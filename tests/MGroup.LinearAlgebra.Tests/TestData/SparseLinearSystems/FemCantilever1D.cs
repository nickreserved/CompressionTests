namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    using Compression.src.MGroup.Solvers.Multigrid;
    using global::MGroup.LinearAlgebra.Matrices;
    using global::MGroup.LinearAlgebra.Matrices.Builders;
    using global::MGroup.LinearAlgebra.Vectors;
    using global::MGroup.MSolve.Discretization.Meshes.Structured;
    using System;
    using System.Linq;

    public class FemCantilever1D : FemCantileverBase
    {
        /// <summary>
        /// Number of nodes per node are 2 displacement and one rotational dofs per node. 3 in total.
        /// </summary>
        public override int NumDofsPerNode { get => 3; }


        /// <summary>
        /// Creates a cantilever Model.
        /// </summary>
        /// <param name="numElementsPerAxis">Array with one entry. The number of elements on cantilever's length axis.</param>
        /// <param name="lengthPerAxis">Array with 3 entries. The entries are cantilever's length (0), height (1) and width (2)</param>
        public FemCantilever1D(int[] numElementsPerAxis, double[] lengthPerAxis)
            : base(new CartesianMesh1D(numElementsPerAxis, lengthPerAxis),
                  lengthPerAxis[1] * lengthPerAxis[2] / 12 * lengthPerAxis[1] * lengthPerAxis[1]) {}

        protected override Matrix ElementStiffness() => throw new NotImplementedException();
        protected override double[] CalcKnownDisplacementsForNode(double[] coords) => throw new NotImplementedException();

        public override IGeometricMultigridModel GenerateModel(int[] numElementsPerAxis, double[] lengthPerAxis)
        { return new FemCantilever1D(numElementsPerAxis, lengthPerAxis);  }
    }
}
