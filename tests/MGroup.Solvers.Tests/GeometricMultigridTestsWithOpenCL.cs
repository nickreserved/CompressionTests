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
    public class GeometricMultigridTestsWithOpenCL
    {
        [Fact]
        public void OpenCLTestInitializationAndRun()
        {
            Platform[] platforms = Platform.GetPlatforms();
            Xunit.Assert.NotEmpty(platforms);
            Device[] devices = platforms[0].GetDevices();     // select a platform to get devices
            Xunit.Assert.NotEmpty(devices);
            OpenCL context = new(platforms[0].platformId, /*devices[0].deviceId*/devices.Select(x => x.deviceId).ToArray());

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

        private static String logFilePath = "out.txt";

        private static void Solve(OpenCL context, Device device, IGeometricMultigridModel model, bool DuVi, bool gaussSeidel, int iterations, double convergenceTolerance, int depth, int iterationsPerLevel)
        {
            Stopwatch stopwatch = new();
            stopwatch.Restart();
            IOpenCLGeometricMultigridSolver solver = DuVi
                ? OpenCLDuViGeometricMultigridSolver.CreateDeepV(device, context, model, gaussSeidel, iterations, true, convergenceTolerance, depth, iterationsPerLevel)
                : OpenCLCsrGeometricMultigridSolver.CreateDeepV(device, context, model, gaussSeidel, iterations, true, convergenceTolerance, depth, iterationsPerLevel);
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
            File.AppendAllText(logFilePath, $"\tMethod: {(gaussSeidel ? "Gauss-Seidel" : "Jacobi")}\n");
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

        private static readonly int[] ElementsPerAxis1 = { 256, 16 };
        private static readonly int[] ElementsPerAxis2 = { 256, 16, 16 };
        private static readonly int[] ElementsPerAxis3 = { 512, 16, 32 };
        private static readonly int[] ElementsPerAxis4 = { 4096, 256 };
        private static readonly double[] LengthPerAxis = { 20, 1, 1 };

        public static IEnumerable<object[]> CantileverData =>
            new List<object[]>
            {
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 1, 1 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  1, 1 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 1, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  1, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 1, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  1, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 1, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  1, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 1, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  1, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 1, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  1, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 1, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  1, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 1, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  1, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 1, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  1, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 2, 1 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  2, 1 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 2, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  2, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 2, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  2, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 2, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  2, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 2, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  2, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 2, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  2, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 2, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  2, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 2, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  2, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 2, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  2, 8 },

                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 1, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  1, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 1, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  1, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 1, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  1, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 1, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  1, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 1, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  1, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 1, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  1, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 1, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  1, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 1, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  1, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 1, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  1, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 2, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  2, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 2, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  2, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 2, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  2, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 2, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  2, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 2, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  2, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 2, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  2, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 2, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  2, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 2, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  2, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 2, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  2, 8 },

                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 2, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  2, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 2, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  2, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 2, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  2, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 2, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  2, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 2, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  2, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 2, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  2, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 2, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  2, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 2, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  2, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 2, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  2, 8 },

                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 3, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  3, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 3, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  3, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 3, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  3, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 3, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  3, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 3, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  3, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 3, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  3, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 3, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  3, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 3, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  3, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 3, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  3, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 4, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  4, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 4, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  4, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 4, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  4, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 4, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  4, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 4, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  4, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 4, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  4, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 4, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  4, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 4, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  4, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 4, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  4, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 6, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  6, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 6, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  6, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 6, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  6, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 6, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  6, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 6, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  6, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 6, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  6, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 6, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  6, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 6, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  6, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 6, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  6, 8 },
            };

        [Theory]
        [MemberData(nameof(CantileverData))]
        public static void CheckCantileverSolutionDeepVWithOpenCL(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    bool GaussSeidel, bool DuVi,
                                                                    int depth = 2, int iterationsPerLevel = 4,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            Platform[] platforms = Platform.GetPlatforms();
            Xunit.Assert.NotEmpty(platforms);
            Device[] devices = platforms[0].GetDevices();     // select a platform to get devices
            Xunit.Assert.NotEmpty(devices);
            //Assert.True(devices[0].extensions.Contains("cl_khr_non_uniform_work_group")); // local_workgroup must be 1 because extension is not supported in my PC
            //OpenCL context = new OpenCL(platforms[0].platformId, devices.Select(x => x.deviceId).ToArray());
            OpenCL context = new(platforms[0].platformId, devices[0].deviceId);

            IGeometricMultigridModel model = elementsPerAxis.Length == 3
                ? new FemCantilever3D(elementsPerAxis, lengthPerAxis)
                : new FemCantilever2D(elementsPerAxis, lengthPerAxis);

            Solve(context, devices[0], model, DuVi, GaussSeidel, iterations, convergenceTolerance, depth, iterationsPerLevel);
        }
    }
}
