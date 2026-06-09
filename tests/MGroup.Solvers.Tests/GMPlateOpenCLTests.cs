using Compression.src.MGroup.Solvers.Multigrid;
using Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems;
using Xunit;

namespace Compression.tests.MGroup.Solvers.Tests
{
    public class GMPlateOpenCLTests
    {
        [Theory]
        [MemberData(
            nameof(GMPlateTests.PlateDataGM),
            MemberType = typeof(GMPlateTests)
        )]
        public static void CheckPlateSolutionDeepVWithOpenCL(int[] elementsPerAxis, double[] lengthPerAxis,
                                                                    bool GaussSeidel, bool DuVi,
                                                                    int depth = 2, int iterationsPerLevel = 4)
        {
            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis,
                                GMPlateTests.ElasticityModulus, GMPlateTests.PoissonRatio, GMPlateTests.DistributedLoad);
            GMCantileverOpenCLTests.CheckSolutionDeepVWithOpenCL(model, GaussSeidel, DuVi, depth, iterationsPerLevel);
        }

        [Theory]
        [MemberData(
            nameof(GMCantileverTests.CantileverDataCG),
            MemberType = typeof(GMCantileverTests)
        )]
        public static void CheckPlateSolutionCGWithOpenCL(int[] elementsPerAxis, double[] lengthPerAxis)
        {
            IGeometricMultigridModel model = new FemPlate(elementsPerAxis, lengthPerAxis,
                                GMPlateTests.ElasticityModulus, GMPlateTests.PoissonRatio, GMPlateTests.DistributedLoad);
            GMCantileverOpenCLTests.CheckSolutionCGWithOpenCL(model);
        }
    }
}
