using CASS.OpenCL;
using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Vectors;
using MGroup.OCL;
using System.Diagnostics;
using Xunit;

namespace Compression.tests.MGroup.Solvers.Tests
{
    public class GeometricMultigridPlateOpenCLTests
    {
        private readonly static string logFilePath = "out_plate.txt";

        private static readonly int[] ElementsPerAxis1 = { 256, 16 };
        private static readonly int[] ElementsPerAxis2 = { 256, 16, 16 };
        private static readonly int[] ElementsPerAxis3 = { 512, 16, 32 };
        private static readonly int[] ElementsPerAxis4 = { 4096, 256 };
        private static readonly double[] LengthPerAxis = { 20, 1, 1 };

        public static IEnumerable<object[]> PlateDataGM =>
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
        [MemberData(nameof(PlateDataGM))]
        public static void CheckPlateSolutionDeepVWithOpenCL(int[] elementsPerAxis, double[] lengthPerAxis,
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

            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis);
            Device device = devices[0];

            Stopwatch stopwatch = new();
            stopwatch.Restart();
            IOpenCLGeometricMultigridSolver solver = DuVi
                ? OpenCLDuViGeometricMultigridSolver.CreateDeepV(devices[0], context, model, GaussSeidel, iterations, true, convergenceTolerance, depth, iterationsPerLevel)
                : OpenCLCsrGeometricMultigridSolver.CreateDeepV(devices[0], context, model, GaussSeidel, iterations, true, convergenceTolerance, depth, iterationsPerLevel);
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
            nameof(GeometricMultigridCantileverBeamTests.CantileverDataCG),
            MemberType = typeof(GeometricMultigridCantileverBeamTests)
        )]
        public static void CheckPlateSolutionCGWithOpenCL(int[] elementsPerAxis, double[] lengthPerAxis)
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

            Device device = devices[0];
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
    }
}
