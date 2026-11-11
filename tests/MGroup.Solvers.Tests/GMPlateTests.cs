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
        public static void CheckPlateSolutionV()
        {
            OutputPlateInfo(ElementsPerAxis1, LengthPerAxis);
            GMCantileverTests.CheckSolutionV(new FemPlate(ElementsPerAxis1, LengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad));
        }
     

        internal static readonly double ElasticityModulus = 52416;
        internal static readonly double PoissonRatio = 0.3;
        internal static readonly double DistributedLoad = -10;

        private static readonly int[] ElementsPerAxis1 = { 48, 32 };
        private static readonly int[] ElementsPerAxis2 = { 256, 128 };
        private static readonly int[] ElementsPerAxis3 = { 512, 384 };
        private static readonly int[] ElementsPerAxis4 = { 768, 640 };
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
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 6, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  6, 1 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 6, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  6, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 6, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  6, 2 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 6, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  6, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 6, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  6, 4 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 6, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  6, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 6, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  6, 6 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, false, 6, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, false, true,  6, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  false, 6, 8 },
                new object[] { ElementsPerAxis2, LengthPerAxis, true,  true,  6, 8 },

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
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 3, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  3, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 3, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  3, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 3, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  3, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 3, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  3, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 3, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  3, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 3, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  3, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 3, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  3, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 3, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  3, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 3, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  3, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 4, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  4, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 4, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  4, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 4, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  4, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 4, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  4, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 4, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  4, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 4, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  4, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 4, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  4, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 4, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  4, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 4, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  4, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 5, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  5, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 5, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  5, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 5, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  5, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 5, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  5, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 5, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  5, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 5, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  5, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 5, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  5, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 5, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  5, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 5, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  5, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 6, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  6, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 6, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  6, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 6, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  6, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 6, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  6, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 6, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  6, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 6, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  6, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 6, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  6, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 6, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  6, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 6, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  6, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 7, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  7, 1 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 7, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  7, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 7, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  7, 2 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 7, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  7, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 7, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  7, 4 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 7, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  7, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 7, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  7, 6 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, false, 7, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, false, true,  7, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  false, 7, 8 },
                new object[] { ElementsPerAxis3, LengthPerAxis, true,  true,  7, 8 },

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
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 5, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  5, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 5, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  5, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 5, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  5, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 5, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  5, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 5, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  5, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 5, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  5, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 5, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  5, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 5, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  5, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 5, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  5, 8 },
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
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 7, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  7, 1 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 7, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  7, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 7, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  7, 2 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 7, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  7, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 7, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  7, 4 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 7, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  7, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 7, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  7, 6 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, false, 7, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, false, true,  7, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  false, 7, 8 },
                new object[] { ElementsPerAxis4, LengthPerAxis, true,  true,  7, 8 },
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
            JacobiPingPong(GaussSeidel, iterationsPerLevel);
            DivisibleByPowerOf2(elementsPerAxis, depth);
            PossibleOutOfPhysicalMemory(elementsPerAxis, depth);
            OutputPlateInfo(elementsPerAxis, lengthPerAxis);
            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad);
            GMCantileverTests.CheckSolutionDeepV(model, GaussSeidel, DuVi, depth, iterationsPerLevel, iterations, convergenceTolerance);
        }


        public static IEnumerable<object[]> PlateDataCG =>
            new List<object[]>
            {
                new object[] { ElementsPerAxis1, LengthPerAxis },
                new object[] { ElementsPerAxis2, LengthPerAxis },
                new object[] { ElementsPerAxis3, LengthPerAxis },
                new object[] { ElementsPerAxis4, LengthPerAxis }
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
            OutputPlateInfo(elementsPerAxis, lengthPerAxis);
            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad);
            GMCantileverTests.SolveCG(model, iterations, convergenceTolerance); // CG
        }

        internal static void OutputPlateInfo(int[] elementsPerAxis, double[] lengthPerAxis) => File.AppendAllText(GMCantileverOpenCLTests.logFilePath,
                $"\nPLATE\n" +
                $"Dimensions: {lengthPerAxis[0]} x {lengthPerAxis[1]} x {lengthPerAxis[2]}\n" + 
                $"Elements: {elementsPerAxis[0]} x {elementsPerAxis[1]}\n" +                
                $"E: {ElasticityModulus}, v: {PoissonRatio}, p: {DistributedLoad}\n");

        internal static void PossibleOutOfPhysicalMemory(int[] elementsPerAxis, int depth)
        {
            long totalPhysicalMemory = System.GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            long totalEntriesInMemory = totalPhysicalMemory / 8; // max number of 'double' matrix entries in memory 
            long totalRowsInMemory = (long) (Math.Sqrt(totalEntriesInMemory) * 0.8); // max number of matrix rows in memory, with safety factor

            int x = elementsPerAxis[0] >> depth; // elements in depth
            int y = elementsPerAxis[1] >> depth; // elements in depth
            ++x; ++y; // nodes in depth
            long totalFreeDoFs = 3 * x * y - 3 * x - 2 * y + 6;

            Xunit.Assert.False(totalFreeDoFs > totalRowsInMemory, "This test RAM requirement exceeds physical memory");
        }

        internal static void JacobiPingPong(bool GaussSeidel, int iterationsPerLevel)
        {
            if (!GaussSeidel && (iterationsPerLevel & 1) != 0)
                throw new Exception("When Jacobi smoother used iterationsPerLevel must be multiple of 2 because buffer 'ping pong' method used");
        }

        internal static void DivisibleByPowerOf2(int[] elementsPerAxis, int depth)
        {
            for (int i = 0; i < elementsPerAxis.Length; ++i)
                if (((elementsPerAxis[i] >> depth) << depth) != elementsPerAxis[i])
                    throw new Exception("Please fix number of elements per axis");
        }
    }
}
