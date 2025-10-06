using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.ConjugateGradient;
using MGroup.LinearAlgebra.Iterative.Stationary;
using MGroup.LinearAlgebra.Iterative.Stationary.CSR;
using MGroup.LinearAlgebra.Iterative.Termination.Convergence;
using MGroup.LinearAlgebra.Iterative.Termination.Iterations;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Vectors;
using System.Diagnostics;
using Xunit;

namespace Compression.tests.MGroup.Solvers.Tests
{
    public static class GeometricMultigridTests
    {
        private static (CsrMatrix stiffness, Vector b, StationaryIterativeMethod methodGaussSeider) OnlyGaussSeidelInitialize(IStructuredModel model, int iterations, double convergenceTolerance)
        {
            (DokRowMajor A, Vector b) = model.CreateLinearSystem();
            CsrMatrix stiffness = A.BuildCsrMatrix(true);

            StationaryIterativeMethod.Factory builder = new StationaryIterativeMethod.Factory(new GaussSeidelIterationCsr());
            builder.ConvergenceCriterion = new AbsoluteSolutionConvergenceCriterion();
            builder.ConvergenceTolerance = convergenceTolerance;
            builder.MaxIterationsProvider = new FixedMaxIterationsProvider(iterations);
            StationaryIterativeMethod methodGaussSeider = builder.Build();
            return (stiffness, b, methodGaussSeider);
        }

        private static IterativeStatistics OnlyGaussSeidelSolve((CsrMatrix stiffness, Vector b, StationaryIterativeMethod methodGaussSeider) tup, Vector x)
            => tup.methodGaussSeider.Solve(tup.stiffness, tup.b, x);

        [Fact]
        public static void CheckCantilever2dSolutionV()
        {
            double convergenceTolerance = 1e-4;
            int iterations = 100000;
            String logFilePath = "out.txt";

            //FemCantilever2D model = new FemCantilever2D(new int[] { 50, 5 }, new double[] { 20, 1, 1 }); // non power of 2 elements per axis
            IGeometricMultigridModel model = new FemCantilever2D(new int[] { 256, 16 }, new double[] { 20, 1, 1 });

            Stopwatch stopwatch = new Stopwatch();

            stopwatch.Restart();
            GeometricMultigridSolver solver = GeometricMultigridSolver.createSimpleV(model, false, GeometricMultigridSolver.MatrixType.CSR, iterations, convergenceTolerance);
            stopwatch.Stop();
            double timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            (IterativeStatistics stats, double[] time) = solver.Solve(x);
            stopwatch.Stop();
            double timeGMGS = stopwatch.Elapsed.TotalMilliseconds;

            File.AppendAllText(logFilePath, $"\n\n\n\n\nRequired time for Geometric Multigrid: {timeGMGI+timeGMGS}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            for (int i = 0; i < time.Length; ++i)
                File.AppendAllText(logFilePath, $"\tLevel {i}: {time[i]}ms\n");

            Assert.True(stats.HasConverged);

            stopwatch.Restart();
            var gs = OnlyGaussSeidelInitialize(model, iterations, convergenceTolerance);
            stopwatch.Stop();
            double timeGSI = stopwatch.Elapsed.TotalMilliseconds;
            x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            stats = OnlyGaussSeidelSolve(gs, x);
            stopwatch.Stop();
            double timeGSS = stopwatch.Elapsed.TotalMilliseconds;
            File.AppendAllText(logFilePath, $"\n\nRequired time for Gauss-Seidel: {timeGSI+timeGSS}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGSI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGSS}ms");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
        }














        private static (CsrMatrix stiffness, Vector b, CGAlgorithm methodGaussSeider) ConjugateGradientInitialize(IStructuredModel model, int iterations, double convergenceTolerance)
        {
            (DokRowMajor A, Vector b) = model.CreateLinearSystem();
            CsrMatrix stiffness = A.BuildCsrMatrix(true);

            CGAlgorithm.Builder builder = new CGAlgorithm.Builder();
            builder.ResidualTolerance = convergenceTolerance;
            builder.MaxIterationsProvider = new FixedMaxIterationsProvider(iterations);
            CGAlgorithm methodCG = builder.Build();
            return (stiffness, b, methodCG);
        }

        private static String logFilePath = "out.txt";

        private static void Solve(IGeometricMultigridModel model, bool gaussSeidel,
            GeometricMultigridSolver.MatrixType matType, int iterations, double convergenceTolerance)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Restart();
            GeometricMultigridSolver solver = GeometricMultigridSolver.createDeepV(model, gaussSeidel, matType, iterations, convergenceTolerance, 2, 4);
            stopwatch.Stop();
            double timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            (IterativeStatistics stats, double[] time) = solver.Solve(x);
            stopwatch.Stop();
            double timeGMGS = stopwatch.Elapsed.TotalMilliseconds;

            File.AppendAllText(logFilePath, $"\nRequired time for Geometric Multigrid: {timeGMGI + timeGMGS}ms\n");
            File.AppendAllText(logFilePath, $"\tMethod: {(gaussSeidel ? "Gauss-Seidel" : "Jacobi")}\n");
            File.AppendAllText(logFilePath, $"\tMatrix type: {(matType == GeometricMultigridSolver.MatrixType.CSR ? "CSR" : "DuVi")}\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            for (int i = 0; i < time.Length; ++i)
                File.AppendAllText(logFilePath, $"\tLevel {i}: {time[i]}ms\n");

            Assert.True(stats.HasConverged);
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
            IterativeStatistics stats = methodCG.Solve(stiffness, b, x, true);  // Solve
            stopwatch.Stop();
            double timeCG = stopwatch.Elapsed.TotalMilliseconds;
            File.AppendAllText(logFilePath, $"\nRequired time for CG with matrix type CSR: {timeCGI + timeCG}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeCGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeCG}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");

        }

        [Fact]
        public static void CheckCantilever2dSolutionDeepV()
        {
            double convergenceTolerance = 1e-5;
            int iterations = 10000;
            IGeometricMultigridModel model = new FemCantilever2D(new int[] { 256, 16 }, new double[] { 20, 1, 1 });

            Solve(model, true, GeometricMultigridSolver.MatrixType.CSR, iterations, convergenceTolerance);
            Solve(model, true, GeometricMultigridSolver.MatrixType.DU_VI, iterations, convergenceTolerance);
            Solve(model, false, GeometricMultigridSolver.MatrixType.CSR, iterations, convergenceTolerance);
            Solve(model, false, GeometricMultigridSolver.MatrixType.DU_VI, iterations, convergenceTolerance);
            Solve(model, iterations, convergenceTolerance);
        }
    }
}
