using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using Xunit;


namespace Compression.tests.MGroup.Solvers.Tests
{
    public static class GMPlateTests
    {
        [Fact]
        public static void CheckPlateSolutionV() => GMCantileverTests.CheckSolutionV(new FemPlate(ElementsPerAxis1, LengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad));
     

        internal static readonly double ElasticityModulus = 52416;
        internal static readonly double PoissonRatio = 0.3;
        internal static readonly double DistributedLoad = -10;

        private static readonly int[] ElementsPerAxis1 = { 48, 32 };
        private static readonly int[] ElementsPerAxis2 = { 512, 512 };
        private static readonly double[] LengthPerAxis = { 3, 2, 1 };

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
        public static void CheckPlateSolutionDeepV(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    bool GaussSeidel, bool DuVi,
                                                                    int depth = 2, int iterationsPerLevel = 4,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad);
            GMCantileverTests.CheckSolutionDeepV(model, GaussSeidel, DuVi, depth, iterationsPerLevel, iterations, convergenceTolerance);
        }


        public static IEnumerable<object[]> PlateCG =>
            new List<object[]>
            {
                new object[] { ElementsPerAxis1, LengthPerAxis },
                new object[] { ElementsPerAxis2, LengthPerAxis }
            };
        [Theory]
        [MemberData(nameof(PlateCG))]
        public static void CheckCantileverSolutionCG(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    int iterations = 2000, double convergenceTolerance = 1e-5)
        {
            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis, ElasticityModulus, PoissonRatio, DistributedLoad);
            GMCantileverTests.Solve(model, iterations, convergenceTolerance); // CG
        }
    }
}
