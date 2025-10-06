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
    public class OpenCLDuViGeometricMultigridSolver : IGeometricMultigridSolver
    {
        private struct DuViMat
        {
            public int[] RowToColumnIndices;
            public int[] RowToDistances;
            public int[] ColumnIndices;
            public byte[] Distances;
            public ushort[] ValueIndices;
            public double[] Values;
        }

        private bool[] LevelDown;
        private readonly int[] LevelIterations;
        private int firstLevel;
        private int totalLevels;

        private int[]? LevelDoFs;
        private bool[] UseLocalMemory;
        private SizeT[][] GlobalWorkSize;
        private SizeT[][] LocalWorkSize;

        private readonly bool GaussSeidel;

        private LdlSkyline? coarseStiffnessLdlFactorized;

        private readonly OpenCL context;
        private CLProgram[] program;

        private CLKernel kernelJacobiInitial;
        private CLKernel[] kernelResidual, kernelResidualWithCheck, kernelJacobi, kernelGaussSeidel, kernelMatrixVectorProduct;

        private CLCommandQueue commandQueue;

        private CLMem[] bufferOfPrecond, bufferOfRowOffsetsToColumns, bufferOfRowOffsetsToDistances, bufferOfColumnIndices, bufferOfDistances, bufferOfValueIndices, bufferOfValues, bufferOfVectorB, bufferOfVectorX;
        private CLMem bufferOfVectorR, bufferOfZero;

        private int[] ElementsOfBufferOfValues; // CHECK: change

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
        public static OpenCLDuViGeometricMultigridSolver CreateSimpleV(Device device, OpenCL context, IGeometricMultigridModel model, bool GaussSeidel = true,
            int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int levelIterations = 4)
        {
            OpenCLDuViGeometricMultigridSolver a = new(context, maxCircleIterations, convergenceTolerance, new bool[] { true, false }, new int[] { levelIterations }, GaussSeidel);
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
        public static OpenCLDuViGeometricMultigridSolver CreateDeepV(Device device, OpenCL context, IGeometricMultigridModel model, bool GaussSeidel = true,
            int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int depth = 2, int levelIterations = 4)
        {
            bool[] levelDown = Enumerable.Repeat(true, depth).Concat(Enumerable.Repeat(false, depth)).ToArray();
            OpenCLDuViGeometricMultigridSolver a = new(context, maxCircleIterations, convergenceTolerance, levelDown, new int[] { levelIterations }, GaussSeidel);
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
        public OpenCLDuViGeometricMultigridSolver(OpenCL context, int maxCircleIterations, double convergenceTolerance, bool[] levelDown, int[] levelIterations, bool GaussSeidel = true)
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
           // NonUniformWorkgroup = device.extensions.Contains("cl_khr_non_uniform_work_group");
            uint LocalWorkgroupSize = Math.Min(device.workgroupSizeMax, (uint) device.workItemSizes[0]);
            uint LocalMemorySize = (uint) device.memSizeLocal;

            GeometricMultigridSolver.MakePath(ref totalLevels, ref LevelDown, ref firstLevel);


            // Generate coarser models
            DuViMat[] mat = new DuViMat[(totalLevels - 1) * 3];
            Vector[] preconditioners = new Vector[totalLevels - 1];
            ElementsOfBufferOfValues = new int[(totalLevels - 1) * 3];
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
                mat[3 * i + 0] = FromDokRowMajor(A);
                mat[3 * i + 1] = FromDokRowMajor(restrictionB);
                mat[3 * i + 2] = FromDokRowMajor(interpolationB);
                (A, _) = model.CreateLinearSystem();

            }
            // initialize number of elements of matrices for copy from global to local memory
            for (int i = 0; i < mat.Length; ++i)
                ElementsOfBufferOfValues[i] = mat[i].Values.Length;

            // level will span kernel with local memory usage (it fits?) or with global (slower)?
            UseLocalMemory = new bool[mat.Length];
            for (int i = 0; i < mat.Length; ++i)
                UseLocalMemory[i] = mat[i].Values.Length * sizeof(double) <= LocalMemorySize;

            coarseStiffnessLdlFactorized = GeometricMultigridSolver.HealJacobiAndCreateLDL(true, A, preconditioners);

            OpenCLCsrGeometricMultigridSolver.CalculateWorkgroupSizes(LevelDoFs, LocalWorkgroupSize, false, GlobalWorkSize, LocalWorkSize);

            // Initialize OpenCL
            program = new CLProgram[2];
            kernelJacobi = new CLKernel[2];
            kernelGaussSeidel = new CLKernel[2];
            kernelResidual = new CLKernel[2];
            kernelResidualWithCheck = new CLKernel[2];
            kernelMatrixVectorProduct = new CLKernel[2];

            program[0] = Program.CreateProgram(context, "DuViGeometricMultigrid", "-cl-std=CL2.0");
            program[1] = Program.CreateProgram(context, "DuViGeometricMultigrid", "-cl-std=CL2.0 -DUSE_LOCAL_MEMORY");

            for (int i = 0; i < 2; ++i)
            {
                kernelJacobi             [i] = context.CreateKernel(program[i], "jacobi_iteration");
                kernelGaussSeidel        [i] = context.CreateKernel(program[i], "gauss_seidel_iteration");
                kernelResidual           [i] = context.CreateKernel(program[i], "residual");
                kernelResidualWithCheck  [i] = context.CreateKernel(program[i], "residual_with_check");
                kernelMatrixVectorProduct[i] = context.CreateKernel(program[i], "matrix_vector_product");
            }
            kernelJacobiInitial = context.CreateKernel(program[0], "jacobi_initial_iteration");

            commandQueue = context.CreateCommandQueue(context.Devices[0]);


            // Reserve and write OpenCL buffers
            // ... for matrices
            bufferOfRowOffsetsToColumns   = new CLMem[mat.Length];
            bufferOfRowOffsetsToDistances = new CLMem[mat.Length];
            bufferOfColumnIndices         = new CLMem[mat.Length];
            bufferOfDistances             = new CLMem[mat.Length];
            bufferOfValueIndices          = new CLMem[mat.Length];
            bufferOfValues                = new CLMem[mat.Length];
            for (int i = 0; i < mat.Length; ++i)
            {
                bufferOfRowOffsetsToColumns[i]   = context.CreateBuffer(CLMemFlags.ReadOnly, mat[i].RowToColumnIndices.Length * sizeof(UInt32));
                bufferOfRowOffsetsToDistances[i] = context.CreateBuffer(CLMemFlags.ReadOnly, mat[i].RowToDistances    .Length * sizeof(UInt32));
                bufferOfColumnIndices[i]         = context.CreateBuffer(CLMemFlags.ReadOnly, mat[i].ColumnIndices     .Length * sizeof(UInt32));
                bufferOfDistances[i]             = context.CreateBuffer(CLMemFlags.ReadOnly, mat[i].Distances         .Length * sizeof(byte));
                bufferOfValueIndices[i]          = context.CreateBuffer(CLMemFlags.ReadOnly, mat[i].ValueIndices      .Length * sizeof(UInt16));
                bufferOfValues[i]                = context.CreateBuffer(CLMemFlags.ReadOnly, mat[i].Values            .Length * sizeof(double));
                context.WriteBuffer(commandQueue, bufferOfRowOffsetsToColumns[i],   CLBool.True, 0, mat[i].RowToColumnIndices.Length * sizeof(UInt32), mat[i].RowToColumnIndices);
                context.WriteBuffer(commandQueue, bufferOfRowOffsetsToDistances[i], CLBool.True, 0, mat[i].RowToDistances    .Length * sizeof(UInt32), mat[i].RowToDistances);
                context.WriteBuffer(commandQueue, bufferOfColumnIndices[i],         CLBool.True, 0, mat[i].ColumnIndices     .Length * sizeof(UInt32), mat[i].ColumnIndices);
                context.WriteBuffer(commandQueue, bufferOfDistances[i],             CLBool.True, 0, mat[i].Distances         .Length * sizeof(byte),   mat[i].Distances);
                context.WriteBuffer(commandQueue, bufferOfValueIndices[i],          CLBool.True, 0, mat[i].ValueIndices      .Length * sizeof(UInt16), mat[i].ValueIndices);
                context.WriteBuffer(commandQueue, bufferOfValues[i],                CLBool.True, 0, mat[i].Values            .Length * sizeof(double), mat[i].Values);
            }
            // ...for vectors
            bufferOfVectorB = new CLMem[totalLevels];
            bufferOfVectorX = new CLMem[totalLevels - 1];
            bufferOfPrecond = new CLMem[totalLevels - 1];
            bufferOfVectorR = context.CreateBuffer(CLMemFlags.ReadWrite, (mat[0].RowToColumnIndices.Length - 1) * sizeof(double));
            for (int i = 0; i < totalLevels - 1; ++i)
            {
                int total = (mat[3 * i].RowToColumnIndices.Length - 1) * sizeof(double);
                bufferOfVectorB[i] = context.CreateBuffer(i == 0 ? CLMemFlags.ReadOnly : CLMemFlags.ReadWrite, total);
                bufferOfVectorX[i] = context.CreateBuffer(                               CLMemFlags.ReadWrite, total);
                bufferOfPrecond[i] = context.CreateBuffer(                               CLMemFlags.ReadOnly,  total);
                context.WriteBuffer(commandQueue, bufferOfPrecond[i], CLBool.True, 0, total, preconditioners[i].RawData);
            }
            bufferOfVectorB[totalLevels - 1] = context.CreateBuffer(CLMemFlags.ReadWrite, (mat[3 * totalLevels - 5].RowToColumnIndices.Length - 1) * sizeof(double));
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

            //context.SetKernelArg(kernelJacobi, 0, bufferOfPrecond                  [currentLevel]);
            //context.SetKernelArg(kernelJacobi, 1, bufferOfRowOffsetsToColumns  [3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 2, bufferOfRowOffsetsToDistances[3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 3, bufferOfColumnIndices        [3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 4, bufferOfDistances            [3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 5, bufferOfValueIndices         [3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 6, bufferOfValues               [3 * currentLevel]);
            //context.SetKernelArg(kernelJacobi, 7, bufferOfVectorB                  [currentLevel]);
            //context.SetKernelArg(kernelJacobi, 8, bufferOfVectorX[currentLevel] OR bufferOfVectorR);
            //context.SetKernelArg(kernelJacobi, 9, bufferOfVectorR               OR bufferOfVectorX[currentLevel]);
            //context.SetKernelArg(kernelJacobi, 10, LevelDoFs                       [currentLevel]);
            // USE_LOCAL_MEMORY
            //context.SetKernelArg(kernelJacobi, 11, ElementsOfBufferOfValues    [3 * currentLevel]);
            //ThrowCLException(OpenCLDriver.clSetKernelArg(
            //                     kernelJacobi, 12, ElementsOfBufferOfValues    [3 * currentLevel] * sizeof(double), 0));

            //context.SetKernelArg(kernelGaussSeidel, 0, bufferOfPrecond                  [currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 1, bufferOfRowOffsetsToColumns  [3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 2, bufferOfRowOffsetsToDistances[3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 3, bufferOfColumnIndices        [3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 4, bufferOfDistances            [3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 5, bufferOfValueIndices         [3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 6, bufferOfValues               [3 * currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 7, bufferOfVectorB                  [currentLevel]);
            //context.SetKernelArg(kernelGaussSeidel, 8, bufferOfVectorX                  [currentLevel]);
            context.SetKernelArg(kernelGaussSeidel[0], 9, bufferOfVectorR);
            context.SetKernelArg(kernelGaussSeidel[1], 9, bufferOfVectorR);
            //context.SetKernelArg(kernelGaussSeidel, 10, LevelDoFs                       [currentLevel]);
            // USE_LOCAL_MEMORY
            //context.SetKernelArg(kernelGaussSeidel, 11, ElementsOfBufferOfValues    [3 * currentLevel]);
            //ThrowCLException(OpenCLDriver.clSetKernelArg(
            //                     kernelGaussSeidel, 12, ElementsOfBufferOfValues    [3 * currentLevel] * sizeof(double), 0));

            //context.SetKernelArg(kernelMatrixVectorProduct, 0, bufferOfRowOffsetsToColumns  [3 * currentLevel] OR bufferOfRowOffsetsToColumns  [3 * currentLevel + 1] OR bufferOfRowOffsetsToColumns  [3 * currentLevel - 1] OR bufferOfRowOffsetsToColumns  [3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 1, bufferOfRowOffsetsToDistances[3 * currentLevel] OR bufferOfRowOffsetsToDistances[3 * currentLevel + 1] OR bufferOfRowOffsetsToDistances[3 * currentLevel - 1] OR bufferOfRowOffsetsToDistances[3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 2, bufferOfColumnIndices        [3 * currentLevel] OR bufferOfColumnIndices        [3 * currentLevel + 1] OR bufferOfColumnIndices        [3 * currentLevel - 1] OR bufferOfColumnIndices        [3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 3, bufferOfDistances            [3 * currentLevel] OR bufferOfDistances            [3 * currentLevel + 1] OR bufferOfDistances            [3 * currentLevel - 1] OR bufferOfDistances            [3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 4, bufferOfValueIndices         [3 * currentLevel] OR bufferOfValueIndices         [3 * currentLevel + 1] OR bufferOfValueIndices         [3 * currentLevel - 1] OR bufferOfValueIndices         [3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 5, bufferOfValues               [3 * currentLevel] OR bufferOfValues               [3 * currentLevel + 1] OR bufferOfValues               [3 * currentLevel - 1] OR bufferOfValues               [3 * currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 6, bufferOfVectorX                  [currentLevel] OR bufferOfVectorR                                     OR bufferOfVectorX                  [currentLevel]     OR bufferOfVectorB                  [currentLevel]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 7, bufferOfVectorR                                 OR bufferOfVectorB                  [currentLevel + 1] OR bufferOfVectorX                  [currentLevel - 1] OR bufferOfVectorX                  [currentLevel - 1]);
            //context.SetKernelArg(kernelMatrixVectorProduct, 8, (byte) 1                                        OR (byte) 1                                            OR (byte) 0                                            OR (byte) 0);
            //context.SetKernelArg(kernelMatrixVectorProduct, 9, LevelDoFs                        [currentLevel] OR LevelDoFs                        [currentLevel + 1] OR LevelDoFs                        [currentLevel - 1] OR LevelDoFs                        [currentLevel - 1]);
            // USE_LOCAL_MEMORY
            //context.SetKernelArg(kernelMatrixVectorProduct, 10, ElementsOfBufferOfValues    [3 * currentLevel] OR ElementsOfBufferOfValues     [3 * currentLevel + 1] OR ElementsOfBufferOfValues     [3 * currentLevel - 1] OR ElementsOfBufferOfValues     [3 * currentLevel - 1]);
            //ThrowCLException(OpenCLDriver.clSetKernelArg(
            //                     kernelMatrixVectorProduct, 11, (ElementsOfBufferOfValues   [3 * currentLevel] OR ElementsOfBufferOfValues     [3 * currentLevel + 1] OR ElementsOfBufferOfValues     [3 * currentLevel - 1] OR ElementsOfBufferOfValues     [3 * currentLevel - 1]) * sizeof(double), 0);

            for (int i = 0; i < 2; ++i)
            {
                //context.SetKernelArg(kernelResidual[i], 0, bufferOfRowOffsetsToColumns  [3 * currentLevel]);
                //context.SetKernelArg(kernelResidual[i], 1, bufferOfRowOffsetsToDistances[3 * currentLevel]);
                //context.SetKernelArg(kernelResidual[i], 2, bufferOfColumnIndices        [3 * currentLevel]);
                //context.SetKernelArg(kernelResidual[i], 3, bufferOfDistances            [3 * currentLevel]);
                //context.SetKernelArg(kernelResidual[i], 4, bufferOfValueIndices         [3 * currentLevel]);
                //context.SetKernelArg(kernelResidual[i], 5, bufferOfValues               [3 * currentLevel]);
                //context.SetKernelArg(kernelResidual[i], 6, bufferOfVectorB                  [currentLevel]);
                //context.SetKernelArg(kernelResidual[i], 7, bufferOfVectorX                  [currentLevel]);
                context.SetKernelArg(kernelResidual[i], 8, bufferOfVectorR);
                //context.SetKernelArg(kernelResidual[i], 9, LevelDoFs                        [currentLevel]);
            }
            // USE_LOCAL_MEMORY
            //context.SetKernelArg(kernelResidual[1], 10, ElementsOfBufferOfValues    [3 * currentLevel]);
            //ThrowCLException(OpenCLDriver.clSetKernelArg(
            //                     kernelResidual[1], 11, ElementsOfBufferOfValues    [3 * currentLevel] * sizeof(double), 0));

            for (int i = 0; i < 2; ++i)
            {
                context.SetKernelArg(kernelResidualWithCheck[i], 0, bufferOfRowOffsetsToColumns  [0]);
                context.SetKernelArg(kernelResidualWithCheck[i], 1, bufferOfRowOffsetsToDistances[0]);
                context.SetKernelArg(kernelResidualWithCheck[i], 2, bufferOfColumnIndices        [0]);
                context.SetKernelArg(kernelResidualWithCheck[i], 3, bufferOfDistances            [0]);
                context.SetKernelArg(kernelResidualWithCheck[i], 4, bufferOfValueIndices         [0]);
                context.SetKernelArg(kernelResidualWithCheck[i], 5, bufferOfValues               [0]);
                context.SetKernelArg(kernelResidualWithCheck[i], 6, bufferOfVectorB              [0]);
                context.SetKernelArg(kernelResidualWithCheck[i], 7, bufferOfVectorX              [0]);
                context.SetKernelArg(kernelResidualWithCheck[i], 8, bufferOfVectorR);
                context.SetKernelArg(kernelResidualWithCheck[i], 9, ConvergenceTolerance);
                context.SetKernelArg(kernelResidualWithCheck[i], 10, bufferOfZero);
                context.SetKernelArg(kernelResidualWithCheck[i], 11, LevelDoFs                   [0]);
            }
            // USE_LOCAL_MEMORY
            context.SetKernelArg(kernelResidualWithCheck[1], 12, ElementsOfBufferOfValues        [0]);
            ThrowCLException(OpenCLDriver.clSetKernelArg(
                                 kernelResidualWithCheck[1], 13, ElementsOfBufferOfValues        [0] * sizeof(double), 0));

#if DEBUG
            //OutputMatrix(mat, LevelDoFs);
            File.Delete(GeometricMultigridSolver.GetLogPath(true, GaussSeidel, true));
#endif
        }


        private static Matrix DuVi2Mat(DuViMat A, int numRows, int numColumns)
        {
            Matrix B = Matrix.CreateZero(numRows, numColumns);
            for (int row = 0; row < numRows; ++row)
            {
                int colBegin = A.RowToColumnIndices[row];
                int colEnd = A.RowToColumnIndices[row + 1];
                int dstIdx = A.RowToDistances[row];
                for (int j = colBegin; j < colEnd; ++j)
                {
                    int column = A.ColumnIndices[j];
                    for (int blkTotal = A.Distances[dstIdx]; blkTotal > 0; --blkTotal)
                    {
                        B[row, column] = A.Values[A.ValueIndices[dstIdx]];
                        column += 1 + A.Distances[++dstIdx];
                    }
                    B[row, column] = A.Values[A.ValueIndices[dstIdx]];
                }
            }
            return B;
        }
        private static void OutputMatrix(DuViMat[] A, int[] LevelDoFs)
        {  
            #if DEBUG
            string path = "duvi_matrices.txt";
            File.Delete(path);
            for (int i = 0; i < A.Length; ++i)
            {
                int j = i % 3;
                int k = i / 3;
                int rows, columns;
                if (j == 0) rows = columns = LevelDoFs[k];
                else if (j == 1) { rows = LevelDoFs[k + 1]; columns = LevelDoFs[k]; }
                else { rows = LevelDoFs[k]; columns = LevelDoFs[k + 1]; }
                if (j == 0)
                {
                    Matrix B = DuVi2Mat(A[i], rows, columns);
                    string line = j == 0 ? "stiffness" : j == 1 ? "restriction" : "interpolation";
                    line += "_matrix(" + k + ") = [" + Environment.NewLine;
                    for (int m = 0; m < B.NumRows; ++m)
                        line += string.Join(" ", B.GetRow(m).RawData.Select(e => e.ToString("G6"))) + Environment.NewLine;
                    line += "]" + Environment.NewLine + Environment.NewLine;
                    //line += "_matrix(" + k + ") = [" + string.Join(Environment.NewLine, B.RawData.Select(e => e.ToString("G6"))) + "]" + Environment.NewLine;
                    File.AppendAllText(path, line);
                }
            }
            #endif
        }

        private void OutputVectorX(int currentLevel, CLMem[] bufferOfVector, string name)
            => OutputVectorX(currentLevel, bufferOfVector[currentLevel], name);
        private void OutputVectorX(int currentLevel, CLMem bufferOfVector, string name)
            => OpenCLCsrGeometricMultigridSolver.OutputVectorX(currentLevel, bufferOfVector, name, true, context, commandQueue, GaussSeidel, LevelDoFs);

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
                        int vidx = currentLevel - 1;
                        int midx = 3 * currentLevel - 1;
                        bool lm = UseLocalMemory[midx];
                        CLKernel kMVP = kernelMatrixVectorProduct[lm ? 1 : 0];
                        context.SetKernelArg(kMVP, 0, bufferOfRowOffsetsToColumns  [midx]);
                        context.SetKernelArg(kMVP, 1, bufferOfRowOffsetsToDistances[midx]);
                        context.SetKernelArg(kMVP, 2, bufferOfColumnIndices        [midx]);
                        context.SetKernelArg(kMVP, 3, bufferOfDistances            [midx]);
                        context.SetKernelArg(kMVP, 4, bufferOfValueIndices         [midx]);
                        context.SetKernelArg(kMVP, 5, bufferOfValues               [midx]);
                        context.SetKernelArg(kMVP, 6, bufferOfVectorB              [currentLevel]);
                        context.SetKernelArg(kMVP, 7, bufferOfVectorX              [vidx]);
                        context.SetKernelArg(kMVP, 8, (byte) 0);
                        context.SetKernelArg(kMVP, 9, LevelDoFs                    [vidx]);
                        if (lm)
                        {
                            context.SetKernelArg(kMVP, 10, ElementsOfBufferOfValues[midx]);
                            ThrowCLException(OpenCLDriver.clSetKernelArg(
                                                 kMVP, 11, ElementsOfBufferOfValues[midx] * sizeof(double), 0));
                        }
                        context.NDRangeKernel(commandQueue, kMVP, 1, null, GlobalWorkSize[vidx], LocalWorkSize[vidx]);

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
                            int midx = 3 * currentLevel;
                            bool lm = UseLocalMemory[midx];
                            CLKernel kGaussSeidel = kernelGaussSeidel[lm ? 1 : 0];
                            context.SetKernelArg(kGaussSeidel, 0, bufferOfPrecond              [currentLevel]);
                            context.SetKernelArg(kGaussSeidel, 1, bufferOfRowOffsetsToColumns  [midx]);
                            context.SetKernelArg(kGaussSeidel, 2, bufferOfRowOffsetsToDistances[midx]);
                            context.SetKernelArg(kGaussSeidel, 3, bufferOfColumnIndices        [midx]);
                            context.SetKernelArg(kGaussSeidel, 4, bufferOfDistances            [midx]);
                            context.SetKernelArg(kGaussSeidel, 5, bufferOfValueIndices         [midx]);
                            context.SetKernelArg(kGaussSeidel, 6, bufferOfValues               [midx]);
                            context.SetKernelArg(kGaussSeidel, 7, bufferOfVectorB              [currentLevel]);
                            context.SetKernelArg(kGaussSeidel, 8, bufferOfVectorX              [currentLevel]);
                            context.SetKernelArg(kGaussSeidel,10, LevelDoFs                    [currentLevel]);
                            if (lm)
                            {
                                context.SetKernelArg(kGaussSeidel, 11, ElementsOfBufferOfValues[midx]);
                                ThrowCLException(OpenCLDriver.clSetKernelArg(
                                                     kGaussSeidel, 12, ElementsOfBufferOfValues[midx] * sizeof(double), 0));
                            }

                            // Gauss-Seidel iterations
                            for (int i = 0; i < lvlIter; ++i)
                                context.NDRangeKernel(commandQueue, kGaussSeidel, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);
                        }
                        else
                        {
                            int i;  // how many Jacobi iterations happen until now

                            // normal Jacobi non-changed-per-iteration parameters
                            int midx = 3 * currentLevel;
                            bool lm = UseLocalMemory[midx];
                            CLKernel kJacobi = kernelJacobi[lm ? 1 : 0];
                            context.SetKernelArg(kJacobi, 0, bufferOfPrecond              [currentLevel]);
                            context.SetKernelArg(kJacobi, 1, bufferOfRowOffsetsToColumns  [midx]);
                            context.SetKernelArg(kJacobi, 2, bufferOfRowOffsetsToDistances[midx]);
                            context.SetKernelArg(kJacobi, 3, bufferOfColumnIndices        [midx]);
                            context.SetKernelArg(kJacobi, 4, bufferOfDistances            [midx]);
                            context.SetKernelArg(kJacobi, 5, bufferOfValueIndices         [midx]);
                            context.SetKernelArg(kJacobi, 6, bufferOfValues               [midx]);
                            context.SetKernelArg(kJacobi, 7, bufferOfVectorB              [currentLevel]);
                            // 8 and 9 below
                            context.SetKernelArg(kJacobi,10, LevelDoFs                    [currentLevel]);
                            if (lm)
                            {
                                context.SetKernelArg(kJacobi, 11, ElementsOfBufferOfValues[midx]);
                                ThrowCLException(OpenCLDriver.clSetKernelArg(
                                                     kJacobi, 12, ElementsOfBufferOfValues[midx] * sizeof(double), 0));
                            }

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
                                context.SetKernelArg(kJacobi, 8, bufferOfVectorR);
                                context.SetKernelArg(kJacobi, 9, bufferOfVectorX[currentLevel]);
                                context.NDRangeKernel(commandQueue, kJacobi, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);

                                i = 2;
                            }
                            else i = 0;

                            // Rest of jacobi iterations (or all Jacobi iterations if initial X != 0)
                            for (; i < lvlIter; i += 2) // do not change < to !=
                            {
                                // normal Jacobi has X as initial X. Result as X is R : PING!
                                context.SetKernelArg(kJacobi, 8, bufferOfVectorX[currentLevel]);
                                context.SetKernelArg(kJacobi, 9, bufferOfVectorR);
                                context.NDRangeKernel(commandQueue, kJacobi, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);

                                // normal Jacobi has R as initial X. Result as X is X : PONG!
                                context.SetKernelArg(kJacobi, 8, bufferOfVectorR);
                                context.SetKernelArg(kJacobi, 9, bufferOfVectorX[currentLevel]);
                                context.NDRangeKernel(commandQueue, kJacobi, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);
                            }
                        }

                        if (LevelDown[step])
                        {
                            // calculate fine residual
                            if (currentLevel == 0)
                            {
                                //context.SetKernelArg(kernelResidualWithCheck, 0, bufferOfRowOffsetsToColumns[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 1, bufferOfRowOffsetsToDistances[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 2, bufferOfColumnIndices[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 3, bufferOfDistances[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 4, bufferOfValueIndices[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 5, bufferOfValues[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 6, bufferOfVectorB[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 7, bufferOfVectorX[0]);
                                //context.SetKernelArg(kernelResidualWithCheck, 8, bufferOfVectorR);
                                //context.SetKernelArg(kernelResidualWithCheck, 9, ConvergenceTolerance);
                                //context.SetKernelArg(kernelResidualWithCheck, 10, bufferOfZero);
                                //context.SetKernelArg(kernelResidualWithCheck, 11, LevelDoFs[currentLevel]);
                                // USE_LOCAL_MEMORY
                                //context.SetKernelArg(kernelResidualWithCheck, 12, ElementsOfBufferOfValues[3 * currentLevel]);
                                //ThrowCLException(OpenCLDriver.clSetKernelArg(
                                //                     kernelResidualWithCheck, 13, ElementsOfBufferOfValues[3 * currentLevel] * sizeof(double), 0));
                                context.NDRangeKernel(commandQueue, kernelResidualWithCheck[UseLocalMemory[3 * currentLevel] ? 1 : 0],
                                    1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);

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
                                int midx = 3 * currentLevel;
                                bool lm = UseLocalMemory[midx];
                                CLKernel kResidual = kernelResidual[lm ? 1 : 0];
                                context.SetKernelArg(kResidual, 0, bufferOfRowOffsetsToColumns  [midx]);
                                context.SetKernelArg(kResidual, 1, bufferOfRowOffsetsToDistances[midx]);
                                context.SetKernelArg(kResidual, 2, bufferOfColumnIndices        [midx]);
                                context.SetKernelArg(kResidual, 3, bufferOfDistances            [midx]);
                                context.SetKernelArg(kResidual, 4, bufferOfValueIndices         [midx]);
                                context.SetKernelArg(kResidual, 5, bufferOfValues               [midx]);
                                context.SetKernelArg(kResidual, 6, bufferOfVectorB              [currentLevel]);
                                context.SetKernelArg(kResidual, 7, bufferOfVectorX              [currentLevel]);
                                //context.SetKernelArg(kResidual, 8, bufferOfVectorR);
                                context.SetKernelArg(kResidual, 9, LevelDoFs                    [currentLevel]);
                                if (lm)
                                {
                                    context.SetKernelArg(kResidual, 10, ElementsOfBufferOfValues[midx]);
                                    ThrowCLException(OpenCLDriver.clSetKernelArg(
                                                         kResidual, 11, ElementsOfBufferOfValues[midx] * sizeof(double), 0));
                                }
                                context.NDRangeKernel(commandQueue, kResidual, 1, null, GlobalWorkSize[currentLevel], LocalWorkSize[currentLevel]);
                            }

                            OutputVectorX(currentLevel, bufferOfVectorB, "B");
                            OutputVectorX(currentLevel, bufferOfVectorX, "X");
                            OutputVectorX(currentLevel, bufferOfVectorR, "R");
                          
                            { // block to make local the multiple-times-used variables midx, lm
                                // fine residual to coarse residual
                                int vidx = currentLevel + 1;
                                int midx = 3 * currentLevel + 1;
                                bool lm = UseLocalMemory[midx];
                                CLKernel kMVP = kernelMatrixVectorProduct[lm ? 1 : 0];
                                context.SetKernelArg(kMVP, 0, bufferOfRowOffsetsToColumns[midx]);
                                context.SetKernelArg(kMVP, 1, bufferOfRowOffsetsToDistances[midx]);
                                context.SetKernelArg(kMVP, 2, bufferOfColumnIndices[midx]);
                                context.SetKernelArg(kMVP, 3, bufferOfDistances[midx]);
                                context.SetKernelArg(kMVP, 4, bufferOfValueIndices[midx]);
                                context.SetKernelArg(kMVP, 5, bufferOfValues[midx]);
                                context.SetKernelArg(kMVP, 6, bufferOfVectorR);
                                context.SetKernelArg(kMVP, 7, bufferOfVectorB[vidx]);
                                context.SetKernelArg(kMVP, 8, (byte)1);
                                context.SetKernelArg(kMVP, 9, LevelDoFs[vidx]);
                                if (lm)
                                {
                                    context.SetKernelArg(kMVP, 10, ElementsOfBufferOfValues[midx]);
                                    ThrowCLException(OpenCLDriver.clSetKernelArg(
                                                         kMVP, 11, ElementsOfBufferOfValues[midx] * sizeof(double), 0));
                                }
                                context.NDRangeKernel(commandQueue, kMVP, 1, null, GlobalWorkSize[vidx], LocalWorkSize[vidx]);
                            }
                        }
                        else
                        {
                            int vidx = currentLevel - 1;
                            int midx = 3 * currentLevel - 1;
                            bool lm = UseLocalMemory[midx];
                            CLKernel kMVP = kernelMatrixVectorProduct[lm ? 1 : 0];
                            context.SetKernelArg(kMVP, 0, bufferOfRowOffsetsToColumns  [midx]);
                            context.SetKernelArg(kMVP, 1, bufferOfRowOffsetsToDistances[midx]);
                            context.SetKernelArg(kMVP, 2, bufferOfColumnIndices        [midx]);
                            context.SetKernelArg(kMVP, 3, bufferOfDistances            [midx]);
                            context.SetKernelArg(kMVP, 4, bufferOfValueIndices         [midx]);
                            context.SetKernelArg(kMVP, 5, bufferOfValues               [midx]);
                            context.SetKernelArg(kMVP, 6, bufferOfVectorX              [currentLevel]);
                            context.SetKernelArg(kMVP, 7, bufferOfVectorX              [vidx]);
                            context.SetKernelArg(kMVP, 8, (byte) 0);
                            context.SetKernelArg(kMVP, 9, LevelDoFs                    [vidx]);
                            if (lm)
                            {
                                context.SetKernelArg(kMVP, 10, ElementsOfBufferOfValues[midx]);
                                ThrowCLException(OpenCLDriver.clSetKernelArg(
                                                     kMVP, 11, ElementsOfBufferOfValues[midx] * sizeof(double), 0));
                            }
                            context.NDRangeKernel(commandQueue, kMVP, 1, null, GlobalWorkSize[vidx], LocalWorkSize[vidx]);

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

        private static DuViMat FromDokRowMajor(DokRowMajor A, double tolerance = 1e-10)
        {
            (double[] values, int[] colIndices, int[] rowOffsets) = A.BuildCsrArrays(true);
            DuViMat B = new()
            {
                RowToColumnIndices = new int[rowOffsets.Length],
                RowToDistances = new int[rowOffsets.Length],
                Distances = new byte[values.Length],
                ValueIndices = new ushort[values.Length]
            };
            List<int> columnIndices = new();

            // DU here
            int valueIndex = 0, valueBeginIndex = 0;
            for (int i = 0;; ++i) // break condition inside
            {
                B.RowToColumnIndices[i] = columnIndices.Count;
                B.RowToDistances[i] = valueIndex;
                if (i == A.NumRows) break;
                int fromIndex = rowOffsets[i];
                int toIndex = rowOffsets[i + 1];
                for (int j = fromIndex; j < toIndex; ++j)
                {
                    int blockCount = 0;
                    columnIndices.Add(colIndices[j]);
                    for (; j + 1 < toIndex && blockCount < 255; ++j)
                    {
                        int diff = colIndices[j + 1] - colIndices[j] - 1;
                        if (diff < 256)
                        {
                            ++blockCount;
                            ++valueIndex;
                            B.Distances[valueIndex] = (byte)diff;
                        }
                        else break;
                    }
                    B.Distances[valueBeginIndex] = (byte)blockCount;
                    ++valueIndex;
                    valueBeginIndex = valueIndex;
                }
            }
            B.ColumnIndices = columnIndices.ToArray();

            Dictionary<double, int> map = tolerance > 0 ? new Dictionary<double, int>(new DuViCompressedSparseMatrix.ToleranceComparer(tolerance))
                                                        : new Dictionary<double, int>();
            List<double> uniqueValues = new();

            // VI (value indexing) here
            for (int i = 0; i < values.Length; ++i)
            {
                double v = values[i];
                if (map.TryGetValue(v, out int index))
                    B.ValueIndices[i] = (ushort) index;
                else
                {
                    B.ValueIndices[i] = (ushort)map.Count;
                    uniqueValues.Add(v);
                    map[v] = map.Count;
                }
            }
            B.Values = uniqueValues.ToArray();

            return B;
        }


        /// <summary>
        /// Duplication of library code because it is private inside library.
        /// </summary>
        /// <param name="error">The error code returned from an OpenCL command</param>
        /// <exception cref="OpenCLException">An exception thrown</exception>
        private static void ThrowCLException(CLError error)  { if (error != CLError.Success) throw new OpenCLException(error); }
    }
}
