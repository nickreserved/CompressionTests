using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.Stationary;
using MGroup.LinearAlgebra.Iterative.Stationary.CSR;
using MGroup.LinearAlgebra.Iterative.Termination.Convergence;
using MGroup.LinearAlgebra.Iterative.Termination.Iterations;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization.Entities;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace Compression.tests.MGroup.Solvers.Tests
{
    public static  class GeometricMultigridTests
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
            FemCantilever2D model = new FemCantilever2D(new int[] { 64, 4 }, new double[] { 20, 1, 1 });

            Stopwatch stopwatch = new Stopwatch();

            stopwatch.Restart();
            GeometricMultigridSolver solver = GeometricMultigridSolver.createSimpleV(model, GeometricMultigridSolver.MatrixType.CSR, iterations, convergenceTolerance);
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


    }
}
