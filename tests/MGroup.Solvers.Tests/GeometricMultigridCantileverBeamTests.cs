using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.ConjugateGradient;
using MGroup.LinearAlgebra.Iterative.Termination.Iterations;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Vectors;
using System.Diagnostics;
using Xunit;


namespace Compression.tests.MGroup.Solvers.Tests
{
    public static class GeometricMultigridCantileverBeamTests
    {
        [Fact]
        public static void CheckCantilever2dSolutionV()
        {
            double convergenceTolerance = 1e-4;
            int iterations = 100000;

            //FemCantilever2D model = new FemCantilever2D(new int[] { 50, 5 }, new double[] { 20, 1, 1 }); // non power of 2 elements per axis
            IGeometricMultigridModel model = new FemCantilever2D(new int[] { 256, 16 }, new double[] { 20, 1, 1 });

            Solve(() => GeometricMultigridSolver.CreateSimpleV(model, false, GeometricMultigridSolver.MatrixType.CSR, iterations, false, convergenceTolerance));
            Solve(() => GeometricMultigridSolver.CreateSimpleV(model, false, GeometricMultigridSolver.MatrixType.DUVI, iterations, false, convergenceTolerance));
            Solve(() => GeometricMultigridSolver.CreateSimpleV(model, true, GeometricMultigridSolver.MatrixType.CSR, iterations, false, convergenceTolerance));
            Solve(() => GeometricMultigridSolver.CreateSimpleV(model, true, GeometricMultigridSolver.MatrixType.DUVI, iterations, false, convergenceTolerance));
        }

        private static void Solve(Func<GeometricMultigridSolver> initializer)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Restart();
            GeometricMultigridSolver solver = initializer();
            stopwatch.Stop();
            double timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector? x = null;
            stopwatch.Restart();
            (x, IterativeStatistics stats, double[] time) = solver.Solve(x);
            stopwatch.Stop();
            double timeGMGS = stopwatch.Elapsed.TotalMilliseconds;

            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\nRequired time for Geometric Multigrid: {timeGMGI + timeGMGS}ms\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tTarget machine: CPU with C#\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tMethod: {(solver.GaussSeidel ? "Gauss-Seidel" : "Jacobi")}\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tMatrix type: {(solver.MatType == GeometricMultigridSolver.MatrixType.CSR ? "CSR" : "DuVi")}\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            for (int i = 0; i < time.Length; ++i)
                File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tLevel {i}: {time[i]}ms\n");

            Xunit.Assert.True(stats.HasConverged);
        }

        private static void Solve(IGeometricMultigridModel model, int iterations, double convergenceTolerance)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Restart();

            // Initialization
            (DokRowMajor A, Vector b) = model.CreateLinearSystem();
            CsrMatrix stiffness = A.BuildCsrMatrix(true);
            CGAlgorithm.Builder builder = new CGAlgorithm.Builder();
            builder.ResidualTolerance = convergenceTolerance;
            builder.MaxIterationsProvider = new FixedMaxIterationsProvider(iterations);
            CGAlgorithm methodCG = builder.Build();

            stopwatch.Stop();
            double timeCGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            //TODO: IterativeStatistics stats = methodCG.Solve(stiffness, b, x, true);  // Solve
            /// =============== REIMPLEMENT THE WHEEL FOR DEBUG PURPOSES
            IterativeStatistics stats = new();
            stats.HasConverged = false;
            stats.NumIterationsRequired = iterations;
            {
                Vector M = Vector.CreateFromArray(stiffness.GetDiagonalAsArray());
                for (int i = 0; i < M.Length; ++i)
                    M.RawData[i] = 1 / M.RawData[i];
                
                Vector r = b.Copy();
                r.SubtractIntoThis(stiffness.Multiply(x));
                Vector z = M.MultiplyEntrywise(r);
                Vector p = z.Copy();
                double rz = r.DotProduct(z);

                for (int iteration = 0; iteration < iterations; ++iteration)
                {
                    Vector Ap = stiffness.Multiply(p);
                    double a = rz / p.DotProduct(Ap);
                    x.AddIntoThis(p.Scale(a));
                    r.AddIntoThis(Ap.Scale(-a));

                    bool nobreak = false;
                    for (int i = 0; i < r.Length; ++i)
                        if (Math.Abs(r[i]) > convergenceTolerance) nobreak = true;
                        else stats.ResidualNormRatioEstimation = Math.Abs(r[i]);
                    if (!nobreak) { stats.HasConverged = true; stats.NumIterationsRequired = iteration; break; }

                    z = M.MultiplyEntrywise(r);
                    double rz2 = r.DotProduct(z);
                    double beta = rz2 / rz;
                    p.ScaleIntoThis(beta);
                    p.AddIntoThis(z);
                    rz = rz2;
                }
            }
            /// =============== END REIMPLEMENTATION

            stopwatch.Stop();
            double timeCG = stopwatch.Elapsed.TotalMilliseconds;
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\nRequired time for CG with matrix type CSR: {timeCGI + timeCG}ms\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tTarget machine: CPU with C#\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tMethod: CG\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tMatrix type: CSR\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tInitialization: {timeCGI}ms\n");
            File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tSolve: {timeCG}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");
            else File.AppendAllText(GeometricMultigridCantileverBeamOpenCLTests.logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");

        }

        private static readonly int[] ElementsPerAxis1 = { 256, 16 };
        private static readonly int[] ElementsPerAxis2 = { 256, 16, 16 };
        private static readonly int[] ElementsPerAxis3 = { 512, 16, 32 };
        private static readonly int[] ElementsPerAxis4 = { 4096, 256 };
        private static readonly double[] LengthPerAxis = { 20, 1, 1 };

        public static IEnumerable<object[]> CantileverDataGM =>
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
        [MemberData(nameof(CantileverDataGM))]
        public static void CheckCantileverSolutionDeepV(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    bool GaussSeidel, bool DuVi,
                                                                    int depth = 2, int iterationsPerLevel = 4,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            IGeometricMultigridModel model = elementsPerAxis.Length == 3
                 ? new FemCantilever3D(elementsPerAxis, lengthPerAxis)
                 : new FemCantilever2D(elementsPerAxis, lengthPerAxis);

            GeometricMultigridSolver.MatrixType mat = DuVi ? GeometricMultigridSolver.MatrixType.DUVI : GeometricMultigridSolver.MatrixType.CSR;

            Solve(() => GeometricMultigridSolver.CreateDeepV(model, GaussSeidel, mat, iterations, false, convergenceTolerance, depth, iterationsPerLevel));
            Solve(model, iterations, convergenceTolerance); // CG
        }


        public static IEnumerable<object[]> CantileverDataCG =>
            new List<object[]>
            {
                new object[] { ElementsPerAxis1, LengthPerAxis },
                new object[] { ElementsPerAxis2, LengthPerAxis },
                new object[] { ElementsPerAxis3, LengthPerAxis },
                new object[] { ElementsPerAxis4, LengthPerAxis },
            };
        [Theory]
        [MemberData(nameof(CantileverDataCG))]
        public static void CheckCantileverSolutionCG(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            IGeometricMultigridModel model = elementsPerAxis.Length == 3
                 ? new FemCantilever3D(elementsPerAxis, lengthPerAxis)
                 : new FemCantilever2D(elementsPerAxis, lengthPerAxis);
           Solve(model, iterations, convergenceTolerance); // CG
        }
    }
}
