using CASS.OpenCL;
using CASS.Types;
using Compression.src.MGroup.OCL;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Triangulation;
using MGroup.LinearAlgebra.Vectors;
using MGroup.OCL;
using System.Diagnostics;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public class OpenCLCsrGeometricMultigridSolver : IGeometricMultigridSolver
    {
        private bool[] LevelDown;
        private readonly int[] LevelIterations;
        private int firstLevel;
        private int totalLevels;

        private int[]? LevelDoFs;
        private SizeT[][] GlobalWorkSize;
        private SizeT[][] LocalWorkSize;

        private readonly bool GaussSeidel;

        private LdlSkyline? coarseStiffnessLdlFactorized;

        private readonly OpenCL context;
        private CLProgram program;

        private CLKernel kernelResidual, kernelResidualWithCheck, kernelJacobi, kernelJacobiInitial, kernelGaussSeidel, kernelMatrixVectorProduct;

        private CLCommandQueue commandQueue;

        private CLMem[] bufferOfPrecond, bufferOfRowOffsets, bufferOfColumnIndices, bufferOfValues, bufferOfVectorB, bufferOfVectorX;
        private CLMem bufferOfVectorR, bufferOfZero;

        private const int MaxCircleIterations = 10000;

        public int MaxIterations { get; set; }
        public double ConvergenceTolerance { get; set; }

        /// <summary>
        /// A shallow-V geometric multigrid algorithm with 2 levels of detail.
        /// </summary>
        /// <param name="device">OpenCL device.</param>
        /// <param name="context">OpenCL context.</param>
        /// <param name="model">The geometric multigrid model.</param>
        /// <param name="GaussSeidel">Solve with Jacobi or Gauss-Seidel method.</param>
        /// <param name="maxCircleIterations">Number of iterations for the algorithm. One iteration is a complete circle.</param>
        /// <param name="convergenceTolerance">The tolerance of solution in order to be the problem solved.</param>
        /// <param name="levelIterations">How many Jacobi or Gauss-Seidel iterations will be executed on each step.</param>
        /// <returns>The solver object ready to solve.</returns>
        public static OpenCLCsrGeometricMultigridSolver CreateSimpleV(Device device, OpenCL context, IGeometricMultigridModel model, bool GaussSeidel = true,
            int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int levelIterations = 4)
        {
            OpenCLCsrGeometricMultigridSolver a = new(context, maxCircleIterations, convergenceTolerance, new bool[] { true, false },
                new int[] { levelIterations }, GaussSeidel);
            a.Initialize(device, model);
            return a;
        }

        /// <summary>
        /// A deep-V geometric multigrid algorithm with at least 2 levels of detail.
        /// </summary>
        /// <param name="device">OpenCL device.</param>
        /// <param name="context">OpenCL context.</param>
        /// <param name="model">The geometric multigrid model.</param>
        /// <param name="GaussSeidel">Solve with Jacobi or Gauss-Seidel method.</param>
        /// <param name="maxCircleIterations">Number of iterations for the algorithm. One iteration is a complete circle.</param>
        /// <param name="convergenceTolerance">The tolerance of solution in order to be the problem solved.</param>
        /// <param name="depth">How many steps down the geometric multigrid will go.</param>
        /// <param name="levelIterations">How many Jacobi or Gauss-Seidel iterations will be executed on each step.</param>
        /// <returns>The solver object ready to solve.</returns>
        public static OpenCLCsrGeometricMultigridSolver CreateDeepV(Device device, OpenCL context, IGeometricMultigridModel model, bool GaussSeidel = true,
            int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int depth = 2, int levelIterations = 4)
        {
            bool[] levelDown = Enumerable.Repeat(true, depth).Concat(Enumerable.Repeat(false, depth)).ToArray();
            OpenCLCsrGeometricMultigridSolver a = new(context, maxCircleIterations, convergenceTolerance, levelDown, new int[] { levelIterations }, GaussSeidel);
            a.Initialize(device, model);
            return a;
        }

        /// <summary>
        /// Constructor of solver for OpenCL.
        /// </summary>
        /// <param name="context">OpenCL context.</param>
        /// <param name="maxCircleIterations">Number of iterations for the algorithm. One iteration is a complete circle.</param>
        /// <param name="convergenceTolerance">The tolerance of solution in order to be the problem solved.</param>
        /// <param name="levelDown">If true, next step of geometric multigrid has lower detail. If false, next step has higher detail.</param>
        /// <param name="levelIterations">How many Jacobi or Gauss-Seidel iterations will be executed on each step.
        /// If steps are more than elements of this array, last value is used. If array is null, then 4 is used for every step.</param>
        /// <param name="GaussSeidel">Solve with Jacobi or Gauss-Seidel method.</param>
        public OpenCLCsrGeometricMultigridSolver(OpenCL context, int maxCircleIterations, double convergenceTolerance, bool[] levelDown, int[] levelIterations,
            bool GaussSeidel = true)
        {
            this.context = context;
            this.GaussSeidel = GaussSeidel;
            LevelDown = levelDown;
            LevelIterations = levelIterations;
            MaxIterations = maxCircleIterations;
            ConvergenceTolerance = convergenceTolerance;
        }



        /// <summary>
        /// Initialize Geometric Multigrid solver
        /// </summary>
        /// <param name="device">OpenCL device.</param>
        /// <param name="model">The geometric multigrid model.</param>
        public void Initialize(Device device, IGeometricMultigridModel model)
        {
            // device specific things
            uint LocalWorkgroupSize = Math.Min(device.workgroupSizeMax, (uint)device.workItemSizes[0]);

            GeometricMultigridSolver.MakePath(ref totalLevels, ref LevelDown, ref firstLevel);

            // Generate coarser models
            CsrMatrix[] mat = new CsrMatrix[(totalLevels - 1) * 3];
            Vector[] preconditioners = new Vector[totalLevels - 1];
            GlobalWorkSize = new SizeT[totalLevels][];
            LocalWorkSize = new SizeT[totalLevels][];
            LevelDoFs = new int[totalLevels];
            (DokRowMajor A, Vector b) = model.CreateLinearSystem();
            LevelDoFs[0] = A.NumRows;
            for (int i = 0; i < totalLevels - 1; ++i)
            {
                (model, DokRowMajor restrictionB, DokRowMajor interpolationB) = model.CreateCoarserModelAndSmoothenerMatrices();
                LevelDoFs[i + 1] = restrictionB.NumRows;
                preconditioners[i] = GeometricMultigridSolver.JacobiPreconditioner(A.RawRows);
                mat[3 * i + 0] = A.BuildCsrMatrix(true);                // Stiffness
                mat[3 * i + 1] = restrictionB.BuildCsrMatrix(true);     // Restriction
                mat[3 * i + 2] = interpolationB.BuildCsrMatrix(true);   // Interpolation
                (A, _) = model.CreateLinearSystem();
            }
            coarseStiffnessLdlFactorized = GeometricMultigridSolver.HealJacobiAndCreateLDL(true, A, preconditioners);

            CalculateWorkgroupSizes(LevelDoFs, LocalWorkgroupSize, false, GlobalWorkSize, LocalWorkSize);

            // Initialize OpenCL
            program = Program.CreateProgram(context, "CsrGeometricMultigrid", "-cl-std=CL2.0");

            kernelJacobi              = context.CreateKernel(program, "jacobi_iteration");
            kernelJacobiInitial       = context.CreateKernel(program, "jacobi_initial_iteration");
            kernelGaussSeidel         = context.CreateKernel(program, "gauss_seidel_iteration");
            kernelResidual            = context.CreateKernel(program, "residual");
            kernelResidualWithCheck   = context.CreateKernel(program, "residual_with_check");
            kernelMatrixVectorProduct = context.CreateKernel(program, "matrix_vector_product");

            commandQueue = context.CreateCommandQueue(context.Devices[0]);


            // Reserve and write OpenCL buffers
            // ... for matrices
            bufferOfRowOffsets    = new CLMem[mat.Length];
            bufferOfColumnIndices = new CLMem[mat.Length];
            bufferOfValues        = new CLMem[mat.Length];
            for (int i = 0; i < mat.Length; ++i)
            {
                bufferOfRowOffsets[i]    = context.CreateBuffer(CLMemFlags.ReadOnly,        mat[i].RawRowOffsets.Length * sizeof(UInt32));
                bufferOfColumnIndices[i] = context.CreateBuffer(CLMemFlags.ReadOnly,        mat[i].RawColIndices.Length * sizeof(UInt32));
                bufferOfValues[i]        = context.CreateBuffer(CLMemFlags.ReadOnly,        mat[i].RawValues.Length     * sizeof(double));
                context.WriteBuffer(commandQueue, bufferOfRowOffsets[i],    CLBool.True, 0, mat[i].RawRowOffsets.Length * sizeof(UInt32), mat[i].RawRowOffsets);
                context.WriteBuffer(commandQueue, bufferOfColumnIndices[i], CLBool.True, 0, mat[i].RawColIndices.Length * sizeof(UInt32), mat[i].RawColIndices);
                context.WriteBuffer(commandQueue, bufferOfValues[i],        CLBool.True, 0, mat[i].RawValues.Length     * sizeof(double), mat[i].RawValues);
            }
            // ...for vectors
            bufferOfVectorB = new CLMem[totalLevels];
            bufferOfVectorX = new CLMem[totalLevels - 1];
            bufferOfPrecond = new CLMem[totalLevels - 1];
            bufferOfVectorR = context.CreateBuffer(CLMemFlags.ReadWrite, mat[0].NumRows * sizeof(double));
            for (int i = 0; i < totalLevels - 1; ++i)
            {
                int total = mat[3 * i].NumRows * sizeof(double);
                bufferOfVectorB[i] = context.CreateBuffer(i == 0 ? CLMemFlags.ReadOnly : CLMemFlags.ReadWrite, total);
                bufferOfVectorX[i] = context.CreateBuffer(                               CLMemFlags.ReadWrite, total);
                bufferOfPrecond[i] = context.CreateBuffer(                               CLMemFlags.ReadOnly, total);
                context.WriteBuffer(commandQueue, bufferOfPrecond[i], CLBool.True, 0, total, preconditioners[i].RawData);
            }
            bufferOfVectorB[totalLevels - 1] = context.CreateBuffer(CLMemFlags.ReadWrite, mat[3 * totalLevels - 5].NumRows * sizeof(double));
            context.WriteBuffer(commandQueue, bufferOfVectorB[0], CLBool.True, 0, b.RawData.Length * sizeof(double), b.RawData);
            // ...for scalars
            bufferOfZero = context.CreateBuffer(CLMemFlags.ReadWrite, sizeof(UInt32));
            context.FillBuffer(commandQueue, bufferOfZero, 0, sizeof(UInt32), 1);


            // If we want NON-BLOCKING transfer we can use CLBool.False and then after each context.WriteBuffer we must keep the context.LastEnqueueEvent
            // then we must use clWaitForEvents(num, events) or clGetEventInfo(...) or clSetEventCallback(event, ...)
            // Also clNDRangeKernel() support previous events to be executed -- unfortunatelly not the wrapper version but the C version underneath
            // It starts to be overcomplicated for a test

            // Now the parameters for each kernel follows.
            // What is commented out, takes different params while algorithm executed.
            // This happens on solve()

            //context.SetKernelArg(kernelJacobiInitial, 0, bufferOfPrecond[currentLevel]);
            //context.SetKernelArg(kernelJacobiInitial, 1, bufferOfVectorB[currentLevel]);
            context.SetKernelArg(kernelJacobiInitial, 2, bufferOfVectorR);
            //context.SetKernelArg(kernelJacobiInitial, 3, LevelDoFs      [currentLevel]);

            //context.SetKernelArg(kernelJacobi, 0, bufferOfPrecond          [currentLevel]);
            //context.SetKernelArg(kernelJacobi, 1, bufferOfRowOffsets   [3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 2, bufferOfColumnIndices[3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 3, bufferOfValues       [3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 4, bufferOfVectorB          [currentLevel]);
            //context.SetKernelArg(kernelJacobi, 5, bufferOfVectorX[currentLevel] OR bufferOfVectorR);
            //context.SetKernelArg(kernelJacobi, 6, bufferOfVectorR               OR bufferOfVectorX[currentLevel]);
            //context.SetKernelArg(kernelJacobi, 7, LevelDoFs                [currentLevel]);

            //context.SetKernelArg(kernelGaussSeidel, 0, bufferOfPrecond          [currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 1, bufferOfRowOffsets   [3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 2, bufferOfColumnIndices[3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 3, bufferOfValues       [3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 4, bufferOfVectorB          [currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 5, bufferOfVectorX          [currentLevel]);
            context.SetKernelArg(kernelGaussSeidel, 6, bufferOfVectorR);
            //context.SetKernelArg(kernelGaussSeidel, 7, LevelDoFs                [currentLevel]);

            //context.SetKernelArg(kernelMatrixVectorProduct, 0, bufferOfRowOffsets   [3 * currentLevel] OR bufferOfRowOffsets   [3 * currentLevel + 1] OR bufferOfRowOffsets   [3 * currentLevel - 1] OR bufferOfRowOffsets   [3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 1, bufferOfColumnIndices[3 * currentLevel] OR bufferOfColumnIndices[3 * currentLevel + 1] OR bufferOfColumnIndices[3 * currentLevel - 1] OR bufferOfColumnIndices[3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 2, bufferOfValues       [3 * currentLevel] OR bufferOfValues       [3 * currentLevel + 1] OR bufferOfValues       [3 * currentLevel - 1] OR bufferOfValues       [3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 3, bufferOfVectorX          [currentLevel] OR bufferOfVectorR                             OR bufferOfVectorX          [currentLevel]     OR bufferOfVectorB          [currentLevel]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 4, bufferOfVectorR                         OR bufferOfVectorB          [currentLevel + 1] OR bufferOfVectorX          [currentLevel - 1] OR bufferOfVectorX          [currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 5, (byte) 1                                OR (byte) 1                                    OR (byte) 0                                    OR (byte) 0);
            //context.SetKernelArg(kernelMatrixVectorProduct, 6, LevelDoFs                [currentLevel] OR LevelDoFs                [currentLevel + 1] OR LevelDoFs                [currentLevel - 1] OR LevelDoFs                [currentLevel - 1]);

            //context.SetKernelArg(kernelResidual, 0, bufferOfRowOffsets   [3 * currentLevel]);
            //context.SetKernelArg(kernelResidual, 1, bufferOfColumnIndices[3 * currentLevel]);
            //context.SetKernelArg(kernelResidual, 2, bufferOfValues       [3 * currentLevel]);
            //context.SetKernelArg(kernelResidual, 3, bufferOfVectorB          [currentLevel]);
            //context.SetKernelArg(kernelResidual, 4, bufferOfVectorX          [currentLevel]);
            context.SetKernelArg(kernelResidual, 5, bufferOfVectorR);
            //context.SetKernelArg(kernelResidual, 6, LevelDoFs                [currentLevel]);

            context.SetKernelArg(kernelResidualWithCheck, 0, bufferOfRowOffsets   [0]);
            context.SetKernelArg(kernelResidualWithCheck, 1, bufferOfColumnIndices[0]);
            context.SetKernelArg(kernelResidualWithCheck, 2, bufferOfValues       [0]);
            context.SetKernelArg(kernelResidualWithCheck, 3, bufferOfVectorB      [0]);
            context.SetKernelArg(kernelResidualWithCheck, 4, bufferOfVectorX      [0]);
            context.SetKernelArg(kernelResidualWithCheck, 5, bufferOfVectorR);
            context.SetKernelArg(kernelResidualWithCheck, 6, ConvergenceTolerance);
            context.SetKernelArg(kernelResidualWithCheck, 7, bufferOfZero);
            context.SetKernelArg(kernelResidualWithCheck, 8, LevelDoFs            [0]);

            #if DEBUG
            //OutputMatrix(mat);
            File.Delete(GeometricMultigridSolver.GetLogPath(false, GaussSeidel, true));
            #endif
        }


        private static void OutputMatrix(CsrMatrix[] A)
        {
#if DEBUG
            string path = "csr_matrices_gpu.txt";
            File.Delete(path);
            for (int i = 0; i < A.Length; ++i)
            {
                int j = i % 3;
                int k = i / 3;
                if (j == 0 && k == 1)
                {
                    string line = j == 0 ? "stiffness" : j == 1 ? "restriction" : "interpolation";
                    Matrix B = A[i].CopyToFullMatrix();
                    line += "_matrix(" + k + ") = [" + Environment.NewLine;
                    for (int m = 0; m < B.NumRows; ++m)
                        line += string.Join(" ", B.GetRow(m).RawData.Select(e => e.ToString("G4"))) + Environment.NewLine;
                    line += "]" + Environment.NewLine + Environment.NewLine;
                    //                    line += "_matrix(" + k + ") = [" + string.Join(Environment.NewLine, A[i].CopyToFullMatrix().RawData.Select(e => e.ToString("G6"))) + "]" + Environment.NewLine;
                    File.AppendAllText(path, line);
                }
            }
            #endif
        }

        private void OutputVectorX(int currentLevel, CLMem[] bufferOfVector, string name)
            => OutputVectorX(currentLevel, bufferOfVector[currentLevel], name);
        private void OutputVectorX(int currentLevel, CLMem bufferOfVector, string name)
            => OutputVectorX(currentLevel, bufferOfVector, name, false, context, commandQueue, GaussSeidel, LevelDoFs);

        internal static void OutputVectorX(int currentLevel, CLMem bufferOfVector, string name,
            bool duvi, OpenCL context, CLCommandQueue commandQueue, bool GaussSeidel, int[] LevelDoFs)
        {
            #if DEBUG
            Vector x = Vector.CreateZero(LevelDoFs[currentLevel]);
            context.ReadBuffer(commandQueue, bufferOfVector, CLBool.True, 0, x.Length * sizeof(double), x.RawData);
            string line = name + "(" + currentLevel + ") = [" + string.Join(" ", x.RawData.Select(e => e.ToString("G14"))) + "]" + Environment.NewLine;
            File.AppendAllText(GeometricMultigridSolver.GetLogPath(duvi, GaussSeidel, true), line);
            #endif
        }

        /// <summary>
        /// Solves the geometric multigrid.
        /// </summary>
        /// <param name="xInitialGuess">The initial guess x vector. If initially first level is not 0, then it corresponds to that level (<see cref="firstLevel"/>).
        /// After solving, it has the solution if the problem converges.</param>
        /// <returns>Algorithm statistics and time counts for the solution in each level.</returns>
        public (IterativeStatistics, double[]) Solve(Vector? xInitialGuess)
        {
            if (LevelDoFs == null) throw new InvalidOperationException("You must call Initialize(model) first");

            if (xInitialGuess != null)
            {
                Debug.Assert(xInitialGuess.Length == LevelDoFs[0]);
                context.WriteBuffer(commandQueue, bufferOfVectorX[0], CLBool.True, 0, LevelDoFs[0] * sizeof(double), xInitialGuess.RawData);
            }

            // if first level is not 0
            if (firstLevel > 0) throw new NotImplementedException("firstLevel != 0 is not implementent yet");
            //for (int i = 0; i < firstLevel; ++i)
            //    r[i + 1] = restriction[i].Multiply(r[i]);

            double[] time = new double[totalLevels];
            Stopwatch stopwatch = new();

            // loop of cycles
            for (int currentIteration = 0; ; ++currentIteration)
            {
                int currentLevel = firstLevel;

                // loop of steps inside a cycle
                for (int step = 0; step < LevelDown.Length; ++step)
                {
                    // Multigrid level time count
                    stopwatch.Restart();

                    if (currentLevel == totalLevels - 1)
                    {
                        OutputVectorX(currentLevel, bufferOfVectorB, "B");

                        // Get vector b from OpenCL device,
                        // LDL factorization on b[currentLevel] and result on b[currentLevel] (instead of x[currentLevel] which is not allocated),
                        // Send back to OpenCL device the b[currentLevel].
                        Vector b = Vector.CreateZero(LevelDoFs.Last());
                        context.ReadBuffer(commandQueue, bufferOfVectorB.Last(), CLBool.True, 0, b.Length * sizeof(double), b.RawData);
                        b = coarseStiffnessLdlFactorized.SolveLinearSystem(b);
                        context.WriteBuffer(commandQueue, bufferOfVectorB.Last(), CLBool.True, 0, b.Length * sizeof(double), b.RawData);

                        OutputVectorX(currentLevel, bufferOfVectorB, "X");

                        // interpolate b[currentLevel] (instead of x[currentLevel] which is not allocated) and add to previous value of x[currentLevel - 1]
                        context.SetKernelArg(kernelMatrixVectorProduct, 0, bufferOfRowOffsets   [3 * currentLevel - 1]);
                        context.SetKernelArg(kernelMatrixVectorProduct, 1, bufferOfColumnIndices[3 * currentLevel - 1]);
                        context.SetKernelArg(kernelMatrixVectorProduct, 2, bufferOfValues       [3 * currentLevel - 1]);
                        context.SetKernelArg(kernelMatrixVectorProduct, 3, bufferOfVectorB          [currentLevel]);
                        context.SetKernelArg(kernelMatrixVectorProduct, 4, bufferOfVectorX          [currentLevel - 1]);
                        context.SetKernelArg(kernelMatrixVectorProduct, 5, (byte)0);
                        context.SetKernelArg(kernelMatrixVectorProduct, 6, LevelDoFs                [currentLevel - 1]);
                        context.NDRangeKernel(commandQueue, kernelMatrixVectorProduct, 1, null, GlobalWorkSize[currentLevel - 1], LocalWorkSize[currentLevel - 1]);

                        OutputVectorX(currentLevel - 1, bufferOfVectorX, "X");
                    }
                    else
                    {
                        // try to solve A * xInitialGuess = b
                        int lvlIter = LevelIterations[Math.Min(currentLevel, LevelIterations.Length - 1)];
                        if (GaussSeidel)
                        {
                            // Cases where initial X must be 0
                            bool ld = step == 0 ? LevelDown.Last() : LevelDown[step - 1];
                            if (currentLevel == 0 && xInitialGuess == null || currentLevel != 0 && ld)
                                context.FillBuffer(commandQueue, bufferOfVectorX[currentLevel], 0, LevelDoFs[currentLevel] * sizeof(double), 0.0);

                            // Gauss-Seidel non-changed-per-iteration parameters
                            context.SetKernelArg(kernelGaussSeidel, 0, bufferOfPrecond          [currentLevel]);
                            context.SetKernelArg(kernelGaussSeidel, 1, bufferOfRowOffsets   [3 * currentLevel]);
                            context.SetKernelArg(kernelGaussSeidel, 2, bufferOfColumnIndices[3 * currentLevel]);
                            context.SetKernelArg(kernelGaussSeidel, 3, bufferOfValues       [3 * currentLevel]);
                            context.SetKernelArg(kernelGaussSeidel, 4, bufferOfVectorB          [currentLevel]);
                            context.SetKernelArg(kernelGaussSeidel, 5, bufferOfVectorX          [currentLevel]);
                            //context.SetKernelArg(kernelGaussSeidel, 6, bufferOfVectorR);
                            context.SetKernelArg(kernelGaussSeidel, 7, LevelDoFs                [currentLevel]);

                            // Gauss-Seidel iterations
                            for (int i = 0; i < lvlIter; ++i)
                                context.NDRangeKernel(commandQueue, kernelGaussSeidel, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);
                        }
                        else // Jacobi
                        {
                            int i;  // how many Jacobi iterations happen until now

                            // normal Jacobi non-changed-per-iteration parameters
                            context.SetKernelArg(kernelJacobi, 0, bufferOfPrecond          [currentLevel]);
                            context.SetKernelArg(kernelJacobi, 1, bufferOfRowOffsets   [3 * currentLevel]);
                            context.SetKernelArg(kernelJacobi, 2, bufferOfColumnIndices[3 * currentLevel]);
                            context.SetKernelArg(kernelJacobi, 3, bufferOfValues       [3 * currentLevel]);
                            context.SetKernelArg(kernelJacobi, 4, bufferOfVectorB          [currentLevel]);
                            // 5 and 6 below
                            context.SetKernelArg(kernelJacobi, 7, LevelDoFs                [currentLevel]);

                            // First 2 Jacobi iterations if initial X = 0
                            bool ld = step == 0 ? LevelDown.Last() : LevelDown[step - 1];
                            if (currentLevel == 0 && xInitialGuess == null || currentLevel != 0 && ld)
                            {
                                // initial Jacobi implies initial X = 0. Result as X is R : PING!
                                context.SetKernelArg(kernelJacobiInitial, 0, bufferOfPrecond[currentLevel]);
                                context.SetKernelArg(kernelJacobiInitial, 1, bufferOfVectorB[currentLevel]);
                                //context.SetKernelArg(kernelJacobiInitial, 2, bufferOfVectorR);
                                context.SetKernelArg(kernelJacobiInitial, 3, LevelDoFs      [currentLevel]);
                                context.NDRangeKernel(commandQueue, kernelJacobiInitial, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);

                                // normal Jacobi has R as initial X. Result as X is X : PONG!
                                context.SetKernelArg(kernelJacobi, 5, bufferOfVectorR);
                                context.SetKernelArg(kernelJacobi, 6, bufferOfVectorX[currentLevel]);
                                context.NDRangeKernel(commandQueue, kernelJacobi, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);

                                i = 2;
                            }
                            else i = 0;

                            // Rest of jacobi iterations (or all Jacobi iterations if initial X != 0)
                            for (; i < lvlIter; i += 2) // do not change < to !=
                            {
                                // normal Jacobi has X as initial X. Result as X is R : PING!
                                context.SetKernelArg(kernelJacobi, 5, bufferOfVectorX[currentLevel]);
                                context.SetKernelArg(kernelJacobi, 6, bufferOfVectorR);
                                context.NDRangeKernel(commandQueue, kernelJacobi, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);
                                // normal Jacobi has R as initial X. Result as X is X : PONG!
                                context.SetKernelArg(kernelJacobi, 5, bufferOfVectorR);
                                context.SetKernelArg(kernelJacobi, 6, bufferOfVectorX[currentLevel]);
                                context.NDRangeKernel(commandQueue, kernelJacobi, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);
                            }
                        }

                        if (LevelDown[step])
                        {
                            // calculate fine residual
                            if (currentLevel == 0)
                            {
                                //context.SetKernelArg(kernelResidualWithCheck, 0, bufferOfRowOffsets[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 1, bufferOfColumnIndices[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 2, bufferOfValues[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 3, bufferOfVectorB[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 4, bufferOfVectorX[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 5, bufferOfVectorR);
                                //context.SetKernelArg(kernelResidualWithCheck, 6, ConvergenceTolerance);
                                //context.SetKernelArg(kernelResidualWithCheck, 7, bufferOfZero);
                                //context.SetKernelArg(kernelResidual, 8, LevelDoFs[0]);
                                context.NDRangeKernel(commandQueue, kernelResidualWithCheck, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);

                                xInitialGuess ??= Vector.CreateZero(LevelDoFs[0]); // must be initialized exactly here!

                                UInt32[] con = new UInt32[1];
                                context.ReadBuffer(commandQueue, bufferOfZero, CLBool.True, 0, sizeof(UInt32), con);
                                bool converged = con[0] == 1;
                                bool failed = (con[0] & 2) == 2;  // some numbers become NaN
                                // small residual or exceeded the iteration number
                                if (converged || failed || currentIteration > MaxIterations)
                                {
                                    context.ReadBuffer(commandQueue, bufferOfVectorX[0], CLBool.True, 0, xInitialGuess.Length * sizeof(double), xInitialGuess.RawData);

                                    return (new IterativeStatistics {
                                                NumIterationsRequired = currentIteration,
                                                ConvergenceCriterion = ("dumb text", ConvergenceTolerance),
                                                HasConverged = converged
                                            }, time);
                                }
                                else // not converged
                                    context.FillBuffer(commandQueue, bufferOfZero, 0, sizeof(UInt32), 1);
                            }
                            else
                            {
                                context.SetKernelArg(kernelResidual, 0, bufferOfRowOffsets   [3 * currentLevel]);
                                context.SetKernelArg(kernelResidual, 1, bufferOfColumnIndices[3 * currentLevel]);
                                context.SetKernelArg(kernelResidual, 2, bufferOfValues       [3 * currentLevel]);
                                context.SetKernelArg(kernelResidual, 3, bufferOfVectorB          [currentLevel]);
                                context.SetKernelArg(kernelResidual, 4, bufferOfVectorX          [currentLevel]);
                                //context.SetKernelArg(kernelResidual, 5, bufferOfVectorR);
                                context.SetKernelArg(kernelResidual, 6, LevelDoFs                [currentLevel]);
                                context.NDRangeKernel(commandQueue, kernelResidual, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);
                            }

                            OutputVectorX(currentLevel, bufferOfVectorB, "B");
                            OutputVectorX(currentLevel, bufferOfVectorX, "X");
                            OutputVectorX(currentLevel, bufferOfVectorR, "R");

                            // fine residual to coarse residual
                            context.SetKernelArg(kernelMatrixVectorProduct, 0, bufferOfRowOffsets   [3 * currentLevel + 1]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 1, bufferOfColumnIndices[3 * currentLevel + 1]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 2, bufferOfValues       [3 * currentLevel + 1]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 3, bufferOfVectorR);
                            context.SetKernelArg(kernelMatrixVectorProduct, 4, bufferOfVectorB          [currentLevel + 1]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 5, (byte) 1);
                            context.SetKernelArg(kernelMatrixVectorProduct, 6, LevelDoFs                [currentLevel + 1]);
                            context.NDRangeKernel(commandQueue, kernelMatrixVectorProduct, 1, null, GlobalWorkSize[currentLevel + 1], LocalWorkSize[currentLevel + 1]);
                        }
                        else
                        {
                            context.SetKernelArg(kernelMatrixVectorProduct, 0, bufferOfRowOffsets   [3 * currentLevel - 1]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 1, bufferOfColumnIndices[3 * currentLevel - 1]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 2, bufferOfValues       [3 * currentLevel - 1]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 3, bufferOfVectorX          [currentLevel]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 4, bufferOfVectorX          [currentLevel - 1]);
                            context.SetKernelArg(kernelMatrixVectorProduct, 5, (byte) 0);
                            context.SetKernelArg(kernelMatrixVectorProduct, 6, LevelDoFs                [currentLevel - 1]);
                            context.NDRangeKernel(commandQueue, kernelMatrixVectorProduct, 1, null, GlobalWorkSize[currentLevel - 1], LocalWorkSize[currentLevel - 1]);

                            OutputVectorX(currentLevel, bufferOfVectorX, "X");
                            OutputVectorX(currentLevel - 1, bufferOfVectorX, "X");
                        }
                    }
                    // Multigrid level time count
                    stopwatch.Stop();
                    time[currentLevel] += stopwatch.Elapsed.TotalMilliseconds;

                    currentLevel += LevelDown[step] ? 1 : -1;
                }
            }
        }

        static public void CalculateWorkgroupSizes(int[] LevelDoFs, uint LocalWorkgroupSize, bool NonUniformWorkgroup, SizeT[][] GlobalWorkSize, SizeT[][] LocalWorkSize)
        {
            // initialize number of global and local work-items for kernels                      // CHECK: change
            int totalLevels = LevelDoFs.Length;
            for (int i = 0; i < totalLevels; ++i)
            {
                SizeT g, l;
                if (LevelDoFs[i] <= LocalWorkgroupSize) g = l = LevelDoFs[i];
                else
                {
                    l = LocalWorkgroupSize;
                    g = LevelDoFs[i];
                    if (!NonUniformWorkgroup) g = (g + l - 1) / l * l;
                }
                LocalWorkSize[i] = new SizeT[] { l };
                GlobalWorkSize[i] = new SizeT[] { g };
            }
        }
    }
}
