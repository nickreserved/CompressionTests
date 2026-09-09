using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using Xunit;


namespace Compression.tests.MGroup.Solvers.Tests
{
    public static class GMPlateTests
    {
        [Fact]
        /// <summary>
        /// Geometric Multigrid test for plate.
        /// </summary>
        /// <remarks>
        /// <list type="bullet">
        /// <item>Geometric multigrid with simple V</item>
        /// <item>2 type of matrices: DUVI and CSR</item>
        /// <item>2 type of smoothers: Jacobi and Gauss-Seidel</item>
        /// </list>
        /// </remarks>
        public static void CheckPlateSolutionV() => GMCantileverTests.CheckSolutionV(new FemPlate(ElementsPerAxis1, LengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad));
     

        internal static readonly double ElasticityModulus = 52416;
        internal static readonly double PoissonRatio = 0.3;
        internal static readonly double DistributedLoad = -10;

        private static readonly int[] ElementsPerAxis1 = { 48, 32 };
        private static readonly int[] ElementsPerAxis2 = { 128, 128 };
        private static readonly double[] LengthPerAxis = { 3, 2, 0.1 };

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
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 3, 1 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  3, 1 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 3, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  3, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 3, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  3, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 3, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  3, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 3, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  3, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 3, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  3, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 3, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  3, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 3, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  3, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 3, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  3, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 4, 1 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  4, 1 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 4, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  4, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 4, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  4, 2 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 4, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  4, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 4, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  4, 4 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 4, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  4, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 4, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  4, 6 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, false, 4, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, false, true,  4, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  false, 4, 8 },
                new object[] { ElementsPerAxis1, LengthPerAxis, true,  true,  4, 8 },

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
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 3, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  3, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 3, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  3, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 3, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  3, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 3, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  3, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 3, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  3, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 3, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  3, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 3, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  3, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 3, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  3, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 3, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  3, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 4, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  4, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 4, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  4, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 4, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  4, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 4, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  4, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 4, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  4, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 4, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  4, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 4, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  4, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 4, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  4, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 4, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  4, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 5, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  5, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 5, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  5, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 5, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  5, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 5, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  5, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 5, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  5, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 5, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  5, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 5, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  5, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 5, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  5, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 5, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  5, 8 },
            };

        [Theory]
        [MemberData(nameof(PlateDataGM))]
        /// <summary>
        /// Geometric Multigrid test for plate.
        /// </summary>
        /// <remarks>
        /// <list type="bullet">
        /// <item>Geometric multigrid with deep V of 1 (simple), 2, 3, 4 and 5 lower levels</item>
        /// <item>2 type of matrices: DUVI and CSR</item>
        /// <item>2 type of smoothers: Jacobi and Gauss-Seidel</item>
        /// <item>1, 2, 4, 6 and 8 smoother iterations</item>
        /// </list>
        /// </remarks>
        /// <param name="elementsPerAxis">Number of elements in each axis. An array of 2.</param>
        /// <param name="lengthPerAxis">Dimensions of object. An array of 3.</param>
        /// <param name="GaussSeidel">Type of smoother. Gauss-Seidel or Jacobi.</param>
        /// <param name="DuVi">Type of matrix, DuVi or CSR.</param>
        /// <param name="depth">Depth of Geometric Multigrid.</param>
        /// <param name="iterationsPerLevel">Smoother iterations in any level.</param>
        /// <param name="iterations">Number of maximum iterations (one iteration is the full circle).</param>
        /// <param name="convergenceTolerance">The residual must become smaller than this threshold.</param>
        public static void CheckPlateSolutionDeepV(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    bool GaussSeidel, bool DuVi,
                                                                    int depth = 2, int iterationsPerLevel = 4,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad);
            GMCantileverTests.CheckSolutionDeepV(model, GaussSeidel, DuVi, depth, iterationsPerLevel, iterations, convergenceTolerance);
        }


        public static IEnumerable<object[]> PlateDataCG =>
            new List<object[]>
            {
                new object[] { ElementsPerAxis1, LengthPerAxis },
                new object[] { ElementsPerAxis2, LengthPerAxis }
            };
        [Theory]
        [MemberData(nameof(PlateDataCG))]
        /// <summary>
        /// PCG test for plate and CSR matrix.
        /// </summary>
        /// <param name="elementsPerAxis">Number of elements in each axis. An array of 2.</param>
        /// <param name="lengthPerAxis">Dimensions of object. An array of 3.</param>
        /// <param name="iterations">Number of maximum iterations</param>
        /// <param name="convergenceTolerance">The residual must become smaller than this threshold.</param>
        public static void CheckPlateSolutionCG(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad);
            GMCantileverTests.SolveCG(model, iterations, convergenceTolerance); // CG
        }
    }
}
