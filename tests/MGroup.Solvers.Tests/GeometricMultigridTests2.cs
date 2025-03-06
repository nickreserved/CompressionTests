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
    public static  class GeometricMultigridTests2
    {
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

        private static IterativeStatistics ConjugateGradientSolve((CsrMatrix stiffness, Vector b, CGAlgorithm methodCG) tup, Vector x)
            => tup.methodCG.Solve(tup.stiffness, tup.b, x, true);

        [Fact]
        public static void CheckCantilever2dSolutionDeepV()
        {
            double convergenceTolerance = 1e-5;
            int iterations = 10000;
            String logFilePath = "out.txt";

            IGeometricMultigridModel model = new FemCantilever2D(new int[] { 256, 16 }, new double[] { 20, 1, 1 });

            Stopwatch stopwatch = new Stopwatch();




            stopwatch.Restart();
            GeometricMultigridSolver solver = GeometricMultigridSolver.createDeepV(model, GeometricMultigridSolver.MatrixType.CSR, iterations, convergenceTolerance, 2, 4);
            stopwatch.Stop();
            double timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            Vector x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            (IterativeStatistics stats, double[] time) = solver.Solve(x);
            stopwatch.Stop();
            double timeGMGS = stopwatch.Elapsed.TotalMilliseconds;

            File.AppendAllText(logFilePath, $"\n\n\n\n\nRequired time for Geometric Multigrid (CSR): {timeGMGI+timeGMGS}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            for (int i = 0; i < time.Length; ++i)
                File.AppendAllText(logFilePath, $"\tLevel {i}: {time[i]}ms\n");

            Assert.True(stats.HasConverged);





            stopwatch.Restart();
            solver = GeometricMultigridSolver.createDeepV(model, GeometricMultigridSolver.MatrixType.DU_VI, iterations, convergenceTolerance, 2, 4);
            stopwatch.Stop();
            timeGMGI = stopwatch.Elapsed.TotalMilliseconds;
            x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            (stats, time) = solver.Solve(x);
            stopwatch.Stop();
            timeGMGS = stopwatch.Elapsed.TotalMilliseconds;

            File.AppendAllText(logFilePath, $"\n\n\n\n\nRequired time for Geometric Multigrid (CSR + DuVi): {timeGMGI + timeGMGS}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeGMGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeGMGS}ms\n");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            for (int i = 0; i < time.Length; ++i)
                File.AppendAllText(logFilePath, $"\tLevel {i}: {time[i]}ms\n");

            Assert.True(stats.HasConverged);





            stopwatch.Restart();
            var cg = ConjugateGradientInitialize(model, iterations, convergenceTolerance);
            stopwatch.Stop();
            double timeCGI = stopwatch.Elapsed.TotalMilliseconds;
            x = Vector.CreateZero(model.NumDofsFree);
            stopwatch.Restart();
            stats = ConjugateGradientSolve(cg, x);
            stopwatch.Stop();
            double timeCG = stopwatch.Elapsed.TotalMilliseconds;
            File.AppendAllText(logFilePath, $"\n\nRequired time for CG: {timeCGI+timeCG}ms\n");
            File.AppendAllText(logFilePath, $"\tInitialization: {timeCGI}ms\n");
            File.AppendAllText(logFilePath, $"\tSolve: {timeCG}ms");
            if (stats.HasConverged)
                File.AppendAllText(logFilePath, $"\tCONVERGED after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");
            else File.AppendAllText(logFilePath, $"\tNOT converged after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");
        }


    }
}
