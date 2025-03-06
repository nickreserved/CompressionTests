using CASS.OpenCL;
using CASS.Types;
using Compression.src.MGroup.LinearAlgebra.Iterative.Stationary;
using Compression.src.MGroup.OCL;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.Stationary;
using MGroup.LinearAlgebra.Iterative.Stationary.CSR;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Triangulation;
using MGroup.LinearAlgebra.Vectors;
using MGroup.OCL;
using Microsoft.Testing.Platform.Configurations;
using System.Diagnostics;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public class GeometricMultigridSolverOpenCL
    {
        public enum MatrixType { CSR, DU_VI }

        private IGeometricMultigridModel Model;

        private bool[] LevelDown;
        private int[] LevelIterations;
        private int firstLevel;
        private int totalLevels;

        private MatrixType matType;

        private LdlSkyline coarseStiffnessLdlFactorized;

        private OpenCL context;
        private CLProgram program;
        private CLKernel kernelFirstJacobiDown;
        private CLKernel kernelFirstDown;
        private CLKernel kernelDown;
        private CLKernel kernelLocalMinima;
        private CLKernel kernelGlobalMinima;
        private CLKernel kernelUp;
        private CLKernel kernelLocalMaxima;
        private CLKernel kernelGlobalMaxima;
        private CLCommandQueue commandQueue;
        private CLMem bufferOfRowOffsets;
        private CLMem bufferOfColumnIndices;
        private CLMem bufferOfValues;
        private CLMem bufferOfVectorR;
        private CLMem bufferOfVectorE;
        private CLMem bufferOfFlattenOffsets;
        private CLMem bufferOfInstructions;

        private const int MaxCircleIterations = 10000;

        public int MaxIterations { get; set; }
        public double ConvergenceTolerance { get; set; }

        public static GeometricMultigridSolverOpenCL createSimpleV(OpenCL context, IGeometricMultigridModel model, MatrixType matType = MatrixType.CSR,
                                    int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, byte fineLevelIterations = 4)
        {
            GeometricMultigridSolverOpenCL a = new GeometricMultigridSolverOpenCL(context, model);
            a.Initialize(maxCircleIterations, convergenceTolerance, new bool[] { true, false }, new int[] { fineLevelIterations }, matType);
            return a;
        }

        public static GeometricMultigridSolverOpenCL createDeepV(OpenCL context, IGeometricMultigridModel model, MatrixType matType = MatrixType.CSR,
                           int maxCircleIterations = MaxCircleIterations, double convergenceTolerance = 1e-6, int depth = 2, byte levelIterations = 4)
        {
            GeometricMultigridSolverOpenCL a = new GeometricMultigridSolverOpenCL(context, model);
            byte[][] instructions = new byte[][] {
                Array.Empty<byte>(),
                Enumerable.Repeat(new byte[] { levelIterations, 255 }, depth).SelectMany(x => x).ToArray(),
                Enumerable.Repeat((byte) 0, depth).ToArray()
            };
            a.Initialize(maxCircleIterations, convergenceTolerance, Enumerable.Repeat(true, depth).Concat(Enumerable.Repeat(false, depth)).ToArray(),
                        new int[] { levelIterations }, matType);
            return a;
        }        

        public GeometricMultigridSolverOpenCL(OpenCL context, IGeometricMultigridModel model) { this.context = context; Model = model; }

        /// <summary>
        /// Initialize Geometric Multigrid solver
        /// </summary>
        /// <param name="maxCircleIterations">Number of iterations for the algorithm. One iteration is a complete circle.</param>
        /// <param name="convergenceTolerance">The tolerance of solution in order to be the problem solved.</param>
        /// <param name="levelDown">If true, next step of geometric multigrid has lower detail. If false, next step has higher detail.</param>
        /// <param name="levelIterations">How many Jacobi or Gauss-Seidel iterations will be executed on each step.</param>
        /// <param name="matType">Type of matrix. Either CSR (compressed sparse rows) or DUVI (compressed __units value indexed)</param>
        public void Initialize(int maxCircleIterations, double convergenceTolerance, bool[] levelDown, int[] levelIterations, MatrixType matType)
        {
            this.matType = matType;
            LevelDown = levelDown;
            LevelIterations = levelIterations;
            MaxIterations = maxCircleIterations;
            ConvergenceTolerance = convergenceTolerance;

            // Make the path
            int currentLevel = 0;
            totalLevels = 1;
            for (int i = 0; i < levelDown.Length; ++i)
            {
                if (levelDown[i])
                {
                    ++currentLevel;
                    if (currentLevel == totalLevels) ++totalLevels;
                }
                else if (currentLevel == 0)
                {
                    ++firstLevel;
                    ++totalLevels;
                }
                else --currentLevel;
            }
            if (currentLevel > firstLevel)
                LevelDown = LevelDown.Concat(Enumerable.Repeat(false, currentLevel - firstLevel)).ToArray();
            else if (currentLevel < firstLevel)
                LevelDown = LevelDown.Concat(Enumerable.Repeat(true, firstLevel - currentLevel)).ToArray();

            // Generate coarser models
            IMatrixView[] LevelStiffness = new IMatrixView[totalLevels - 1];
            IMatrixView[] restriction = new IMatrixView[totalLevels - 1];
            IMatrixView[] interpolation = new IMatrixView[totalLevels - 1];
            (DokRowMajor A, Vector b) = Model.CreateLinearSystem();
            IGeometricMultigridModel currentModel = Model;
            for (int i = 0; i < totalLevels; ++i)
            {
                (currentModel, DokRowMajor restrictionB, DokRowMajor interpolationB) = currentModel.CreateCoarserModelAndSmoothenerMatrices();
                switch(matType)
                {
                    case MatrixType.CSR:
                        LevelStiffness[i] = A.BuildCsrMatrix(true);
                        restriction[i] = restrictionB.BuildCsrMatrix(true);
                        interpolation[i] = interpolationB.BuildCsrMatrix(true);
                        break;
                    case MatrixType.DU_VI:
                        LevelStiffness[i] = new DuViCompressedSparseMatrix(A);
                        restriction[i] = new DuViCompressedSparseMatrix(restrictionB);
                        interpolation[i] = new DuViCompressedSparseMatrix(interpolationB);
                        break;
                }
                (A, _) = currentModel.CreateLinearSystem();
            }
            SkylineMatrix coarseStiffness = SkylineMatrix.CreateFromMatrix(A.BuildCsrMatrix(true).CopyToFullMatrix(), 1e-15);
            coarseStiffnessLdlFactorized = coarseStiffness.FactorLdl(true, 1e-15);

            if (matType == MatrixType.DU_VI) throw new NotImplementedException("DU_VI on OpenCL");

            // What is happening here?
            // The CSR matrices of LevelStiffness, Restriction, Interpolation deconstructed to RawRowOffsets, RawValues and RawColumnIndices and
            // serialized to arrays to send to OpenCL kernel. Along with them the vector x and b.
            // The offsets[] array:

            // offsets[ 0] = 0 : The offset of vector R[0], E[0] (R[0] = b, E[0] = xInitial) in bufferOfVectorR and bufferOfVectorE OpenCL buffers
            // offsets[ 1] = 0 : The offset of RawRowOffsets of LevelStiffness[0] in bufferOfRowOffsets OpenCL buffer
            // offsets[ 2] = 0 : The offset of RawValues & RawColumnIndices of LevelStiffness[0] in bufferOfValues & bufferOfColumnIndices OpenCL buffers
            // offsets[ 3]     : The offset of RawRowOffsets of    restriction[0] in bufferOfRowOffsets OpenCL buffer
            // offsets[ 4]     : The offset of RawValues & RawColumnIndices of    restriction[0] in bufferOfValues & bufferOfColumnIndices OpenCL buffers
            // offsets[ 5]     : The offset of RawRowOffsets of  interpolation[0] in bufferOfRowOffsets OpenCL buffer
            // offsets[ 6]     : The offset of RawValues & RawColumnIndices of  interpolation[0] in bufferOfValues & bufferOfColumnIndices OpenCL buffers

            // offsets[ 7]     : The offset of vector R[1], E[1] in bufferOfVectorR and bufferOfVectorE OpenCL buffers
            // offsets[ 8]     : The offset of RawRowOffsets of LevelStiffness[1] in bufferOfRowOffsets OpenCL buffer
            // offsets[ 9]     : The offset of RawValues & RawColumnIndices of LevelStiffness[1] in bufferOfValues & bufferOfColumnIndices OpenCL buffers
            // offsets[10]     : The offset of RawRowOffsets of    restriction[1] in bufferOfRowOffsets OpenCL buffer
            // offsets[11]     : The offset of RawValues & RawColumnIndices of    restriction[1] in bufferOfValues & bufferOfColumnIndices OpenCL buffers
            // offsets[12]     : The offset of RawRowOffsets of  interpolation[1] in bufferOfRowOffsets OpenCL buffer
            // offsets[13]     : The offset of RawValues & RawColumnIndices of  interpolation[1] in bufferOfValues & bufferOfColumnIndices OpenCL buffers

            // etc for index 2, 3, ...

            // offsets.Last(3) : The END (not including) offset of vector R.Last, E.Last in bufferOfVectorR and bufferOfVectorE OpenCL buffers
            // offsets.Last(2) : The END (not including) offset of RawRowOffsets of interpolation.Last in bufferOfRowOffsets OpenCL buffer
            // offsets.Last    : The END (not including) offset of RawValues & RawColumnIndices of interpolation[1] in bufferOfValues & bufferOfColumnIndices OpenCL buffers

            int[] offsets = new int[LevelStiffness.Length * (3 * 2 + 1) + 3];
            offsets[0] = offsets[1] = offsets[2] = 0;
            for (int i = 0; i < LevelStiffness.Length; ++i)
            {
                int j = i * (3 * 2 + 1);
                // Indices of LevelStiffness
                CsrMatrix m = (CsrMatrix)LevelStiffness[i];
                offsets[j + 3] = m.RawRowOffsets.Length + offsets[j + 1];
                offsets[j + 4] = m.RawValues    .Length + offsets[j + 2];
                // Indices of restriction
                m = (CsrMatrix)restriction[i];
                offsets[j + 5] = m.RawRowOffsets.Length + offsets[j + 3];
                offsets[j + 6] = m.RawValues    .Length + offsets[j + 4];
                // Indices of interpolation
                m = (CsrMatrix)interpolation[i];
                offsets[j + 8] = m.RawRowOffsets.Length + offsets[j + 5];
                offsets[j + 9] = m.RawValues    .Length + offsets[j + 6];
                // Indices of vector
                offsets[j + 7] = m.NumRows              + offsets[j + 0]; // only after LevelStiffness or interpolation for NumRows
            }
            int totalVectorElements = offsets[offsets.Length - 3];
            int totalRows =           offsets[offsets.Length - 2];
            int totalValues =         offsets[offsets.Length - 1];

            // offsets of instruction blocks
            instructionOffsets = new int[instructions.Length + 1];
            for (int i = 0; i < instructions.Length; ++i)
                instructionOffsets[i + 1] = instructions[i].Length + instructionOffsets[i];
            int totalInstructions = instructionOffsets.Last();

            // Initialize OpenCL
            program = Program.CreateProgram(context, "HybridGaussSeidel");
            kernel = context.CreateKernel(program, "hybrid_gauss_seidel_step_with_CSR");
            commandQueue = context.CreateCommandQueue(context.Devices[0]);

            // Reserve OpenCL buffers
            bufferOfRowOffsets = context.CreateBuffer(CLMemFlags.ReadOnly, totalRows * sizeof(UInt32));         // Serialized CSR matrices row offsets
            bufferOfColumnIndices = context.CreateBuffer(CLMemFlags.ReadOnly, totalValues * sizeof(UInt32));    // Serialized CSR matrices column indices
            bufferOfValues = context.CreateBuffer(CLMemFlags.ReadOnly, totalValues * sizeof(double));           // Serialized CSR matrices values
            bufferOfVectorR = context.CreateBuffer(CLMemFlags.ReadWrite, totalVectorElements * sizeof(double)); // Values of dense vectors elements
            bufferOfVectorE = context.CreateBuffer(CLMemFlags.ReadWrite, totalVectorElements * sizeof(double)); // Values of dense vectors elements
            bufferOfFlattenOffsets = context.CreateBuffer(CLMemFlags.ReadOnly, offsets.Length * sizeof(UInt32));// Offsets of CSR matrices in serialized RawRowOffsets, RawColIndices & RawValues
            bufferOfInstructions = context.CreateBuffer(CLMemFlags.ReadOnly, totalInstructions * sizeof(byte)); // Instructions

            // Write (Transfer) to OpenCL buffers
            for (int i = 0; i < LevelStiffness.Length; ++i)
            {
                int j = i * (3 * 2 + 1);
                // Copy LevelStiffness RawRowOffsets, RawColIndices & RawValues
                CsrMatrix m = (CsrMatrix)LevelStiffness[i];
                context.WriteBuffer(commandQueue, bufferOfRowOffsets,    CLBool.True, offsets[j + 1] * sizeof(UInt32), offsets[j + 3] * sizeof(UInt32), m.RawRowOffsets);
                context.WriteBuffer(commandQueue, bufferOfColumnIndices, CLBool.True, offsets[j + 2] * sizeof(UInt32), offsets[j + 4] * sizeof(UInt32), m.RawColIndices);
                context.WriteBuffer(commandQueue, bufferOfValues,        CLBool.True, offsets[j + 2] * sizeof(double), offsets[j + 4] * sizeof(double), m.RawValues);
                // Copy restriction RawRowOffsets, RawColIndices & RawValues
                m = (CsrMatrix)restriction[i];
                context.WriteBuffer(commandQueue, bufferOfRowOffsets,    CLBool.True, offsets[j + 3] * sizeof(UInt32), offsets[j + 5] * sizeof(UInt32), m.RawRowOffsets);
                context.WriteBuffer(commandQueue, bufferOfColumnIndices, CLBool.True, offsets[j + 4] * sizeof(UInt32), offsets[j + 6] * sizeof(UInt32), m.RawColIndices);
                context.WriteBuffer(commandQueue, bufferOfValues,        CLBool.True, offsets[j + 4] * sizeof(double), offsets[j + 6] * sizeof(double), m.RawValues);
                // Copy interpolation RawRowOffsets, RawColIndices & RawValues
                m = (CsrMatrix)restriction[i];
                context.WriteBuffer(commandQueue, bufferOfRowOffsets,    CLBool.True, offsets[j + 5] * sizeof(UInt32), offsets[j + 8] * sizeof(UInt32), m.RawRowOffsets);
                context.WriteBuffer(commandQueue, bufferOfColumnIndices, CLBool.True, offsets[j + 6] * sizeof(UInt32), offsets[j + 9] * sizeof(UInt32), m.RawColIndices);
                context.WriteBuffer(commandQueue, bufferOfValues,        CLBool.True, offsets[j + 6] * sizeof(double), offsets[j + 9] * sizeof(double), m.RawValues);
            }
            // Copy offsets of matrices in serialized RawRowOffsets, RawColIndices & RawValues
            context.WriteBuffer(commandQueue, bufferOfFlattenOffsets, CLBool.True, 0, offsets.Length * sizeof(UInt32), offsets);
            // Copy values of vector b (as vector R[0]) -- rest of R vectors calculated from OpenCL kernels 
            context.WriteBuffer(commandQueue, bufferOfVectorR, CLBool.True, offsets[0], offsets[3 * 2 + 1] * sizeof(double), b.RawData);
            // Copy values of vector xInitial (as vector E[0]) -- NOT HERE! It is not available in initialization! Later on solve()

            // If we want NON-BLOCKING transfer we can use CLBool.False and then after each context.WriteBuffer we must keep the context.LastEnqueueEvent
            // then we must use clWaitForEvents(num, events) or clGetEventInfo(...) or clSetEventCallback(event, ...)
            // Also clNDRangeKernel() support previous events to be executed -- unfortunatelly not the wrapper version but the C version underneath
            // It starts to be overcomplicated for a test

            context.SetKernelArg(kernel, 0, bufferOfRowOffsets);
            context.SetKernelArg(kernel, 1, bufferOfColumnIndices);
            context.SetKernelArg(kernel, 2, bufferOfValues);
            context.SetKernelArg(kernel, 4, new double[100]);
        }

        public (IterativeStatistics, double[]) Solve(Vector xInitialGuess)
        {

            /*
            context.WriteBuffer(commandQueue, bufferOfVectorE, CLBool.True, 0, xInitialGuess.Length * sizeof(double), xInitialGuess.RawData);
            context.SetKernelArg(kernel, 3, bufferOfVector);




            context.NDRangeKernel(commandQueue, kernel, 2, new SizeT[] { 0 }, new SizeT[] { 5 }, new SizeT[] { 10 });





            





            IStationaryIteration[] methodGaussSeidel = new IStationaryIteration[LevelStiffness.Length];
            for (int i = 0; i < methodGaussSeidel.Length; ++i)
            {
                methodGaussSeidel[i] = matType == MatrixType.CSR
                    ? new GaussSeidelIterationCsr()
                    : new GaussSeidelIterationCsrDuVi();
                methodGaussSeidel[i].UpdateMatrix(LevelStiffness[i], false);
            }

            Vector[] x = new Vector[LevelStiffness.Length];
            x[0] = xInitialGuess;

            Vector[] r = new Vector[totalLevels];
            r[0] = b;
            // if first level is not 0
            for (int i = 0; i < firstLevel; ++i)
                r[i + 1] = restriction[i].Multiply(r[i]);

            double[] time = new double[totalLevels];
            Stopwatch stopwatch = new Stopwatch();

            // loop of cycles
            for (int iterations = 0; ; ++iterations)
            {
                int currentLevel = firstLevel;

                // loop of steps inside a cycle
                for (int step = 0; step < LevelSteps; ++step)
                {
                    // Multigrid level time count
                    stopwatch.Restart();

                    if (currentLevel == LevelStiffness.Length)
                    {
                        Vector eCoarse = coarseStiffnessLdlFactorized.SolveLinearSystem(r[currentLevel]);
                        Vector eFine = interpolation.Last().Multiply(eCoarse);
                        x[currentLevel - 1].AddIntoThis(eFine);
                    }
                    else
                    {
                        // try to solve A * xInitialGuess = b
                        int lvlIter = LevelIterations[Math.Min(currentLevel, LevelIterations.Length - 1)];
                        for (int i = 0; i < lvlIter; ++i)
                            methodGaussSeidel[currentLevel].Execute(r[currentLevel], x[currentLevel]);

                        if (LevelDown[step])
                        {
                            // calculate residual
                            Vector rFine = r[currentLevel].Copy();
                            rFine.SubtractIntoThis(LevelStiffness[currentLevel].Multiply(x[currentLevel]));

                            // small residual or exceeded the iteration number
                            if (currentLevel == 0)
                            {
                                double residual = rFine.Norm2();
                                bool converged = residual <= ConvergenceTolerance;
                                if (converged || iterations > MaxIterations)
                                {
                                    IterativeStatistics stats = new IterativeStatistics();
                                    stats.NumIterationsRequired = iterations;
                                    stats.ConvergenceCriterion = ("dumb text", residual);
                                    stats.HasConverged = converged;
                                    return (stats, time);
                                }
                            }

                            // fine residual to coarse residual
                            r[currentLevel + 1] = restriction[currentLevel].Multiply(rFine);
                            if (currentLevel < totalLevels - 2)
                                x[currentLevel + 1] = Vector.CreateZero(r[currentLevel + 1].Length);
                        }
                        else
                        {
                            Vector eFine = interpolation[currentLevel - 1].Multiply(x[currentLevel]);
                            x[currentLevel - 1].AddIntoThis(eFine);
                        }
                    }
                    // Multigrid level time count
                    stopwatch.Stop();
                    time[currentLevel] += stopwatch.Elapsed.TotalMilliseconds;

                    currentLevel += LevelDown[step] ? 1 : -1;
                }
            }*/
            return (new IterativeStatistics(), new double[1]);
        }

    }
}
