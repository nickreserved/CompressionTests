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
            Assert.NotEmpty(platforms);
            Device[] devices = platforms[0].GetDevices();     // select a platform to get devices
            Assert.NotEmpty(devices);
            OpenCL context = new OpenCL(platforms[0].platformId, /*devices[0].deviceId*/devices.Select(x => x.deviceId).ToArray());

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
            Assert.True(c);
        }

        private static String logFilePath = "out.txt";

        private static void Solve(OpenCL context, Device device, IGeometricMultigridModel model, bool DuVi, bool gaussSeidel, int iterations, double convergenceTolerance, int depth, int iterationsPerLevel)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Restart();
            IGeometricMultigridSolver solver = DuVi
                ? OpenCLDuViGeometricMultigridSolver.CreateDeepV(device, context, model, gaussSeidel, iterations, convergenceTolerance, depth, iterationsPerLevel)
                : OpenCLCsrGeometricMultigridSolver.CreateDeepV(device, context, model, gaussSeidel, iterations, convergenceTolerance, depth, iterationsPerLevel);
            stopwatch.Stop();
            double timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector? x = null;
            stopwatch.Restart();
            (IterativeStatistics stats, double[] time) = solver.Solve(x);
            stopwatch.Stop();
            double timeGMGS = stopwatch.Elapsed.TotalMilliseconds;
            File.AppendAllText(logFilePath, $"\nRequired time for Geometric Multigrid: {timeGMGI + timeGMGS}ms\n");
            File.AppendAllText(logFilePath, $"\tTarget machine: GPU {device.name}\n");
            File.AppendAllText(logFilePath, $"\tMethod: {(gaussSeidel ? "Gauss-Seidel" : "Jacobi")}\n");
            File.AppendAllText(logFilePath, $"\tMatrix type: {(DuVi ? "DuVi" : "CSR")}\n");
            File.AppendAllText(logFilePath, $"\tDimensions: {model.Mesh.Dimension}\n");
            File.AppendAllText(logFilePath, $"\tFree DoFs: {model.NumDofsFree}\n");
            File.AppendAllText(logFilePath, $"\tDepth of V: {depth}\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
//            for (int i = 0; i < time.Length; ++i)
//                File.AppendAllText(logFilePath, $"\tLevel {i}: {time[i]}ms\n");

            Assert.True(stats.HasConverged);
        }

        private static readonly int[] ElementsPerAxis1 = { 256, 16 };
        private static readonly int[] ElementsPerAxis2 = { 4096, 256 };
        private static readonly int[] ElementsPerAxis3 = { 256, 16, 16 };
        private static readonly int[] ElementsPerAxis4 = { 512, 16, 32 };
        private static readonly double[] LengthPerAxis = { 20, 1, 1 };

        public static IEnumerable<object[]> CantileverData =>
            new List<object[]>
            {
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true, false, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true, true, 2 },

                new object[] { ElementsPerAxis2, LengthPerAxis, false, false },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true },
                new object[] { ElementsPerAxis2, LengthPerAxis, true, false },
                new object[] { ElementsPerAxis2, LengthPerAxis, true, true },

                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true, false, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true, true, 2 },

                new object[] { ElementsPerAxis4, LengthPerAxis, false, false },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true },
                new object[] { ElementsPerAxis4, LengthPerAxis, true, false },
                new object[] { ElementsPerAxis4, LengthPerAxis, true, true },
            };

        [Theory]
        [MemberData(nameof(CantileverData))]
        public static void CheckCantileverSolutionDeepVWithOpenCL(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    bool GaussSeidel, bool DuVi,
                                                                    int depth = 0, int iterationsPerLevel = 4,
                                                                    int iterations = 10000, double convergenceTolerance = 1e-5)
        {
            Platform[] platforms = Platform.GetPlatforms();
            Assert.NotEmpty(platforms);
            Device[] devices = platforms[0].GetDevices();     // select a platform to get devices
            Assert.NotEmpty(devices);
            //Assert.True(devices[0].extensions.Contains("cl_khr_non_uniform_work_group")); // local_workgroup must be 1 because extension is not supported in my PC
            //OpenCL context = new OpenCL(platforms[0].platformId, devices.Select(x => x.deviceId).ToArray());
            OpenCL context = new OpenCL(platforms[0].platformId, devices[0].deviceId);

            if (depth == 0)
            {
                // an approximation of DoFs (multiplication engage elements but DoFs arise from nodes)
                long product = elementsPerAxis.Length * elementsPerAxis.Aggregate(1L, (acc, x) => acc * x);
                while (product > 16384) { product /= 4; ++depth; }
            }


            //IGeometricMultigridModel model = new FemCantilever2D(new int[] { 256, 16 }, new double[] { 20, 1, 1 });
            //IGeometricMultigridModel model = new FemCantilever2D(new int[] { 2048, 128 }, new double[] { 20, 1, 1 });
            IGeometricMultigridModel model = elementsPerAxis.Length == 3
                ? new FemCantilever3D(elementsPerAxis, lengthPerAxis)
                : new FemCantilever2D(elementsPerAxis, lengthPerAxis);
            //IGeometricMultigridModel model = new FemCantilever3D(new int[] { 2048, 128, 256 }, new double[] { 20, 1, 1 });

            Solve(context, devices[0], model, DuVi, GaussSeidel, iterations, convergenceTolerance, depth, iterationsPerLevel);
        }
    }
}
