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
    public static class GMCantileverTests
    {
        // ==================== GM


        /// <summary>
        /// Geometric Multigrid test for cantilever.
        /// </summary>
        /// <remarks>
        /// <list type="bullet">
        /// <item>Cantilever with quad elements</item>
        /// <item>Geometric multigrid with simple V</item>
        /// <item>2 type of matrices: DUVI and CSR</item>
        /// <item>2 type of smoothers: Jacobi and Gauss-Seidel</item>
        /// </list>
        /// </remarks>
        [Fact]
        public static void CheckCantilever2dSolutionV()
        {
            GMCantileverOpenCLTests.OutputCantileverInfo(ElementsPerAxis1, LengthPerAxis);
            CheckSolutionV(new FemCantilever2D(ElementsPerAxis1, LengthPerAxis));
        }

        /// <summary>
        /// Geometric Multigrid test for any type of model.
        /// </summary>
        /// <remarks>
        /// This function is used from both cantilever and plate (or any other addition in the future).
        /// <list type="bullet">
        /// <item>Any type of model</item>
        /// <item>Geometric multigrid with simple V</item>
        /// <item>2 type of matrices: DUVI and CSR</item>
        /// <item>2 type of smoothers: Jacobi and Gauss-Seidel</item>
        /// </list>
        /// </remarks>
        /// <param name="model">The model.</param>
        internal static void CheckSolutionV(IGeometricMultigridModel model)
        {
            double convergenceTolerance = 1e-6;
            int iterations = 100000;

            string log = GMCantileverOpenCLTests.OutputDimDofs(model);

            Solve(() => GeometricMultigridSolver.CreateSimpleV(model, false, GeometricMultigridSolver.MatrixType.CSR, iterations, false, convergenceTolerance), log);
            Solve(() => GeometricMultigridSolver.CreateSimpleV(model, false, GeometricMultigridSolver.MatrixType.DUVI, iterations, false, convergenceTolerance), log);
            Solve(() => GeometricMultigridSolver.CreateSimpleV(model, true, GeometricMultigridSolver.MatrixType.CSR, iterations, false, convergenceTolerance), log);
            Solve(() => GeometricMultigridSolver.CreateSimpleV(model, true, GeometricMultigridSolver.MatrixType.DUVI, iterations, false, convergenceTolerance), log);
        }

        /// <summary>
        /// Solver for Geometric Multigrid with any configuration.
        /// </summary>
        /// <remarks>
        /// This function is used from both cantilever and plate (or any other addition in the future).
        /// </remarks>
        /// <param name="initializer">A lambda expression with no parameters, which returns a GeometricMultigridSolver object.</param>
        /// <param name="log">String with information for output</param>
        internal static void Solve(Func<GeometricMultigridSolver> initializer, string log)
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

            File.AppendAllText(GMCantileverOpenCLTests.logFilePath,
                "Target machine: CPU with C#\n" +
                "Method: Geometric Multigrid\n" +
                $"Smoother: {(solver.GaussSeidel ? "Gauss-Seidel" : "Jacobi")}\n" +
                $"Matrix type: {(solver.MatType == GeometricMultigridSolver.MatrixType.CSR ? "CSR" : "DuVi")}\n" +
                $"Smoother iterations: {solver.LevelSteps}\n" +
                log +
                $"Depth of V: {solver.TotalLevels}\n" +
                $"Initialization: {timeGMGI}ms\n" +
                $"Solve: {timeGMGS}ms\n" +
                $"Total time: {timeGMGI + timeGMGS}ms\n" +
                $"{(stats.HasConverged ? "CONVERGED" : "NOT converged")} after {stats.NumIterationsRequired} iterations and a residual of {stats.ConvergenceCriterion.value}\n");
            //for (int i = 0; i < time.Length; ++i)
            //    File.AppendAllText(GMCantileverOpenCLTests.logFilePath, $"Level {i}: {time[i]}ms\n");

            Xunit.Assert.True(stats.HasConverged);
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

        /// <summary>
        /// Geometric Multigrid test for cantilever.
        /// </summary>
        /// <remarks>
        /// <list type="bullet">
        /// <item>Cantilever with quad elements or hexa elements</item>
        /// <item>Geometric multigrid with deep V of 1 (simple), 2, 3, 4 and 6 lower levels</item>
        /// <item>2 type of matrices: DUVI and CSR</item>
        /// <item>2 type of smoothers: Jacobi and Gauss-Seidel</item>
        /// <item>1, 2, 4, 6 and 8 smoother iterations</item>
        /// </list>
        /// </remarks>
        /// <param name="elementsPerAxis">Number of elements in each axis. An array of 2 or 3.</param>
        /// <param name="lengthPerAxis">Dimensions of object. An array of 3.</param>
        /// <param name="GaussSeidel">Type of smoother. Gauss-Seidel or Jacobi.</param>
        /// <param name="DuVi">Type of matrix, DuVi or CSR.</param>
        /// <param name="depth">Depth of Geometric Multigrid.</param>
        /// <param name="iterationsPerLevel">Smoother iterations in any level.</param>
        /// <param name="iterations">Number of maximum iterations (one iteration is the full circle).</param>
        /// <param name="convergenceTolerance">The residual must become smaller than this threshold.</param>
        [Theory]
        [MemberData(nameof(CantileverDataGM))]
        public static void CheckCantileverSolutionDeepV(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    bool GaussSeidel, bool DuVi,
                                                                    int depth = 2, int iterationsPerLevel = 4,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            GMCantileverOpenCLTests.OutputCantileverInfo(elementsPerAxis, lengthPerAxis);
            IGeometricMultigridModel model = elementsPerAxis.Length == 3
                 ? new FemCantilever3D(elementsPerAxis, lengthPerAxis)
                 : new FemCantilever2D(elementsPerAxis, lengthPerAxis);
            CheckSolutionDeepV(model, GaussSeidel, DuVi, depth, iterationsPerLevel, iterations, convergenceTolerance);
        }

        /// <summary>
        /// Geometric Multigrid test for any type of model.
        /// </summary>
        /// <remarks>
        /// This function is used from both cantilever and plate (or any other addition in the future).
        /// <list type="bullet">
        /// <item>Any type of model.</item>
        /// <item>Geometric multigrid with deep V of 1 (simple), 2, 3, 4 and 6 lower levels</item>
        /// <item>2 type of matrices: DUVI and CSR</item>
        /// <item>2 type of smoothers: Jacobi and Gauss-Seidel</item>
        /// <item>1, 2, 4, 6 and 8 smoother iterations</item>
        /// </list>
        /// </remarks>
        /// <param name="model">The model.</param>
        /// <param name="GaussSeidel">Type of smoother. Gauss-Seidel or Jacobi.</param>
        /// <param name="DuVi">Type of matrix, DuVi or CSR.</param>
        /// <param name="depth">Depth of Geometric Multigrid.</param>
        /// <param name="iterationsPerLevel">Smoother iterations in any level.</param>
        /// <param name="iterations">Number of maximum iterations (one iteration is the full circle).</param>
        /// <param name="convergenceTolerance">The residual must become smaller than this threshold.</param>
        internal static void CheckSolutionDeepV(IGeometricMultigridModel model,
                                                            bool GaussSeidel, bool DuVi,
                                                            int depth = 2, int iterationsPerLevel = 4,
                                                            int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            GeometricMultigridSolver.MatrixType mat = DuVi ? GeometricMultigridSolver.MatrixType.DUVI : GeometricMultigridSolver.MatrixType.CSR;
            Solve(() => GeometricMultigridSolver.CreateDeepV(model, GaussSeidel, mat, iterations, false, convergenceTolerance, depth, iterationsPerLevel),
                GMCantileverOpenCLTests.OutputDimDofs(model));
        }











        // ==================== PCG






        public static IEnumerable<object[]> CantileverDataCG =>
            new List<object[]>
            {
                new object[] { ElementsPerAxis1, LengthPerAxis },
                new object[] { ElementsPerAxis2, LengthPerAxis },
                new object[] { ElementsPerAxis3, LengthPerAxis },
                new object[] { ElementsPerAxis4, LengthPerAxis },
            };

        /// <summary>
        /// PCG test for cantilever with quad or hexa elements and CSR matrix.
        /// </summary>
        /// <param name="elementsPerAxis">Number of elements in each axis. An array of 2 or 3.</param>
        /// <param name="lengthPerAxis">Dimensions of object. An array of 3.</param>
        /// <param name="iterations">Number of maximum iterations</param>
        /// <param name="convergenceTolerance">The residual must become smaller than this threshold.</param>
        [Theory]
        [MemberData(nameof(CantileverDataCG))]
        public static void CheckCantileverSolutionCG(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            GMCantileverOpenCLTests.OutputCantileverInfo(elementsPerAxis, lengthPerAxis);
            IGeometricMultigridModel model = elementsPerAxis.Length == 3
                 ? new FemCantilever3D(elementsPerAxis, lengthPerAxis)
                 : new FemCantilever2D(elementsPerAxis, lengthPerAxis);
           SolveCG(model, iterations, convergenceTolerance); // CG
        }

        /// <summary>
        /// Solves the model with PCG with CSR matrix.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="iterations">Number of maximum iterations allowed before failure.</param>
        /// <param name="convergenceTolerance">A residual must be at most this threshold.</param>
        internal static void SolveCG(IGeometricMultigridModel model, int iterations, double convergenceTolerance)
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
            File.AppendAllText(GMCantileverOpenCLTests.logFilePath,
                "Target machine: CPU with C#\n" +
                "Method: PCG\n" +
                "Matrix type: CSR\n" +
                GMCantileverOpenCLTests.OutputDimDofs(model) +
                $"Initialization: {timeCGI}ms\n" +
                $"Solve: {timeCG}ms\n" +
                $"Total time: {timeCGI + timeCG}ms\n" +
                $"{(stats.HasConverged ? "CONVERGED" : "NOT converged")} after {stats.NumIterationsRequired} iterations and a residual of {stats.ResidualNormRatioEstimation}\n");
        }
    }
}
