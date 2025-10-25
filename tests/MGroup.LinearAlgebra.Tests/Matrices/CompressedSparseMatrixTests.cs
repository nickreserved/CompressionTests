using Compression.tests.MGroup.Solvers.Tests;
using MGroup.Constitutive.Structural;
using MGroup.LinearAlgebra.Commons;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Vectors;
using MGroup.NumericalAnalyzers;
using MGroup.Solvers.Iterative;

using Xunit;

namespace MGroup.LinearAlgebra.Tests.Matrices
{
    public static class CompressedSparseMatrixTests
	{
        [Fact]
		public static void CheckDenseAndSparseToCompressedConversion()
		{
			int width = 1024, height = 2;
			double[] elements = new double[width * height];
			elements[5] = 1; elements[6] = 1; elements[8] = 1; elements[width - 24] = 2; elements[width - 1] = 2;
			elements[1 * width + 50] = 3; elements[1 * width + width - 4] = 3;
			DuViCompressedSparseMatrix mat1 = new DuViCompressedSparseMatrix(height, width, elements, 1E-10);

			int[] colIndex = { 5, 6, 8, width - 24, width - 1,   50, width - 4 };
			double[] values = { 1, 1, 1, 2, 2,    3, 3 };
			int[] rowIndex = { 0, 5, 7 };
			DuViCompressedSparseMatrix mat2 = new DuViCompressedSparseMatrix(height, width, values, colIndex, rowIndex, 1E-10);

			double[] vecData = new double[width];
			Array.Copy(elements, vecData, width);
			Vector vec = Vector.CreateFromArray(vecData);
			
			Vector vec1 = Vector.CreateZero(2), vec2 = Vector.CreateZero(2);

			mat1.Multiply(vec, vec1);
			mat1.Multiply(vec, vec2);

			Xunit.Assert.True(vec1[0] == 11 && vec2[0] == 11 && vec1[1] == 0 && vec2[1] == 0);
		}

        [Fact]
        public static void CheckCantileverSolve()
        {
            CantileverBeam beam = new CantileverBeam.Builder().BuildWithQuad4Elements(100, 10);

            //LibrarySettings.LinearAlgebraProviders = LinearAlgebraProviderChoice.MKL;
            var solverFactory = new PcgSolver2.Factory();
            //solverFactory.DofOrderer = new DofOrderer(new NodeMajorDofOrderingStrategy(), new NullReordering()); // default
            var algebraicModel = solverFactory.BuildAlgebraicModel(beam.Model);
            PcgSolver2 solver = solverFactory.BuildSolver(algebraicModel);

            // Structural problem provider
            var provider = new ProblemStructural(beam.Model, algebraicModel);

            // Linear static analysis
            var childAnalyzer = new LinearAnalyzer(algebraicModel, solver, provider);
            var parentAnalyzer = new StaticAnalyzer(algebraicModel, provider, childAnalyzer);

            // Run the analysis
            parentAnalyzer.Initialize();
            parentAnalyzer.Solve();

            // Check output
            double endDeflectionExpected = beam.CalculateEndDeflectionWithEulerBeamTheory();
            double endDeflectionComputed =
                beam.CalculateAverageEndDeflectionFromSolution(solver.LinearSystem.Solution, algebraicModel);
            var comparer = new ValueComparer(2e-2);
            Xunit.Assert.True(comparer.AreEqual(endDeflectionExpected, endDeflectionComputed));
        }
    }
}
