using CASS.OpenCL;
using CASS.Types;
using Compression.src.MGroup.OCL;
using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Vectors;
using MGroup.OCL;
using System.Diagnostics;
using Xunit;

namespace Compression.tests.MGroup.Solvers.Tests
{
    public class GMCantileverOpenCLTests
    {
        [Fact]
        public void OpenCLTestInitializationAndRun()
        {
            (OpenCL context, _) = InitializeOpenCL();
            CLProgram program = Program.CreateProgram(context, "CsrGeometricMultigrid", "-cl-std=CL2.0");
            CLKernel kernel = context.CreateKernel(program, "matrix_vector_product");
            CLCommandQueue commandQueue = context.CreateCommandQueue(context.Devices[0]);

            int[] rowIndices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
            double[] values = new double[] { 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
            CLMem bufferOfRowIndices = context.CreateBuffer(CLMemFlags.ReadOnly, rowIndices.Length * sizeof(int));
            CLMem bufferOfColumnIndices = context.CreateBuffer(CLMemFlags.ReadOnly, values.Length * sizeof(int));
            CLMem bufferOfValues = context.CreateBuffer(CLMemFlags.ReadOnly, values.Length * sizeof(double));
            CLMem bufferOfVectorX = context.CreateBuffer(CLMemFlags.ReadOnly, values.Length * sizeof(double));
            CLMem bufferOfVectorY = context.CreateBuffer(CLMemFlags.WriteOnly, values.Length * sizeof(double));

            context.WriteBuffer(commandQueue, bufferOfRowIndices, CLBool.True, 0, rowIndices.Length * sizeof(int), rowIndices);
            context.WriteBuffer(commandQueue, bufferOfColumnIndices, CLBool.True, 0, values.Length * sizeof(int), rowIndices);
            context.WriteBuffer(commandQueue, bufferOfValues, CLBool.True, 0, values.Length * sizeof(double), values);
            context.WriteBuffer(commandQueue, bufferOfVectorX, CLBool.True, 0, values.Length * sizeof(double), values);

            context.SetKernelArg(kernel, 0, bufferOfRowIndices);
            context.SetKernelArg(kernel, 1, bufferOfColumnIndices);
            context.SetKernelArg(kernel, 2, bufferOfValues);
            context.SetKernelArg(kernel, 3, bufferOfVectorX);
            context.SetKernelArg(kernel, 4, bufferOfVectorY);
            context.SetKernelArg(kernel, 5, (byte)1);
            context.SetKernelArg(kernel, 6, values.Length);
            context.NDRangeKernel(commandQueue, kernel, 1, null, new SizeT[] { values.Length }, new SizeT[] { values.Length });

            double[] result = new double[values.Length];
            context.ReadBuffer(commandQueue, bufferOfVectorY, CLBool.True, 0, values.Length * sizeof(double), result);
            bool c = true;
            for (int i = 0; i < values.Length; ++i)
                if (result[i] != values[i] * values[i]) { c = false; break; }
            Xunit.Assert.True(c);
        }

        internal static readonly String logFilePath = "out.txt";

        [Theory]
        [MemberData(
            nameof(GMCantileverTests.CantileverDataGM),
            MemberType = typeof(GMCantileverTests)
        )]
        public static void CheckCantileverSolutionDeepVWithOpenCL(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    bool GaussSeidel, bool DuVi,
                                                                    int depth = 2, int iterationsPerLevel = 4)
        {
            IGeometricMultigridModel model = elementsPerAxis.Length == 3
                ? new FemCantilever3D(elementsPerAxis, lengthPerAxis)
                : new FemCantilever2D(elementsPerAxis, lengthPerAxis);
            CheckSolutionDeepVWithOpenCL(model, GaussSeidel, DuVi, depth, iterationsPerLevel);
        }

        internal static void CheckSolutionDeepVWithOpenCL(IGeometricMultigridModel model,
                                                          bool GaussSeidel, bool DuVi,
                                                          int depth = 2, int iterationsPerLevel = 4)
        {
            (OpenCL context, Device device) = InitializeOpenCL();
            int iterations = 20000;
            double convergenceTolerance = 1e-5;

            Stopwatch stopwatch = new();
            stopwatch.Restart();
            IOpenCLGeometricMultigridSolver solver = DuVi
                ? OpenCLDuViGeometricMultigridSolver.CreateDeepV(device, context, model, GaussSeidel, iterations, true, convergenceTolerance, depth, iterationsPerLevel)
                : OpenCLCsrGeometricMultigridSolver.CreateDeepV(device, context, model, GaussSeidel, iterations, true, convergenceTolerance, depth, iterationsPerLevel);
            stopwatch.Stop();
            double timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector? x = null;
            stopwatch.Restart();
            (x, IterativeStatistics stats, double[] time) = solver.Solve(x);
            stopwatch.Stop();
            double timeGMGS = stopwatch.Elapsed.TotalMilliseconds;
            solver.ReleaseOpenCLResources();
            File.AppendAllText(logFilePath, $"\nRequired time for Geometric Multigrid: {timeGMGI + timeGMGS}ms\n");
            File.AppendAllText(logFilePath, $"\tTarget machine: GPU {device.name}\n");
            File.AppendAllText(logFilePath, $"\tMethod: {(GaussSeidel ? "Gauss-Seidel" : "Jacobi")}\n");
            File.AppendAllText(logFilePath, $"\tMatrix type: {(DuVi ? "DuVi" : "CSR")}\n");
            File.AppendAllText(logFilePath, $"\tDimensions: {model.Mesh.Dimension}\n");
            File.AppendAllText(logFilePath, $"\tFree DoFs: {model.NumDofsFree}\n");
            File.AppendAllText(logFilePath, $"\tDepth of V: {depth}\n");
            File.AppendAllText(logFilePath, $"\tSmoother iterations: {iterationsPerLevel}\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            Xunit.Assert.True(stats.HasConverged);
            //File.WriteAllText("result_vector_x.txt", string.Join("\n", x.RawData));
        }

        [Theory]
        [MemberData(
            nameof(GMCantileverTests.CantileverDataCG),
            MemberType = typeof(GMCantileverTests)
        )]
        public static void CheckCantileverSolutionCGWithOpenCL(int[] elementsPerAxis, double[] lengthPerAxis)
        {
            IGeometricMultigridModel model = elementsPerAxis.Length == 3
                ? new FemCantilever3D(elementsPerAxis, lengthPerAxis)
                : new FemCantilever2D(elementsPerAxis, lengthPerAxis);
            CheckSolutionCGWithOpenCL(model);
        }

        internal static void CheckSolutionCGWithOpenCL(IGeometricMultigridModel model)
        {
            (OpenCL context, Device device) = InitializeOpenCL();
            int iterations = 20000;
            double convergenceTolerance = 1e-5;

            Stopwatch stopwatch = new();
            stopwatch.Restart();
            OpenCLCsrPcgSolver solver = new(context, iterations, convergenceTolerance);
            solver.Initialize(device, model);
            stopwatch.Stop();
            double timePCGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector? x = null;
            stopwatch.Restart();
            (x, IterativeStatistics stats) = solver.Solve(x);
            stopwatch.Stop();
            double timePCGS = stopwatch.Elapsed.TotalMilliseconds;
            solver.ReleaseOpenCLResources();
            File.AppendAllText(logFilePath, $"\nRequired time for PCG method with CSR matrix type: {timePCGI + timePCGS}ms\n");
            File.AppendAllText(logFilePath, $"\tTarget machine: GPU {device.name}\n");
            File.AppendAllText(logFilePath, $"\tDimensions: {model.Mesh.Dimension}\n");
            File.AppendAllText(logFilePath, $"\tFree DoFs: {model.NumDofsFree}\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timePCGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timePCGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            Xunit.Assert.True(stats.HasConverged);
            //File.WriteAllText("result_vector_x.txt", string.Join("\n", x.RawData));
        }

        internal static (OpenCL, Device) InitializeOpenCL()
        {
            Platform[] platforms = Platform.GetPlatforms();
            Xunit.Assert.NotEmpty(platforms);
            Device[] devices = platforms[0].GetDevices();     // select a platform to get devices
            Xunit.Assert.NotEmpty(devices);
            //Assert.True(devices[0].extensions.Contains("cl_khr_non_uniform_work_group")); // local_workgroup must be 1 because extension is not supported in my PC
            //OpenCL context = new OpenCL(platforms[0].platformId, devices.Select(x => x.deviceId).ToArray());
            OpenCL context = new(platforms[0].platformId, devices[0].deviceId);
            Device device = devices[0];
            return (context, device);
        }
    }
}
