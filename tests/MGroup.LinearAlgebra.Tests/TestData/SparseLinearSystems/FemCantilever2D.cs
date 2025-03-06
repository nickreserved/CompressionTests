namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    using Compression.src.MGroup.Solvers.Multigrid;
    using global::MGroup.LinearAlgebra.Matrices;
    using global::MGroup.LinearAlgebra.Matrices.Builders;
    using global::MGroup.LinearAlgebra.Vectors;
    using global::MGroup.MSolve.Discretization.Meshes.Structured;
    using System;
    using System.Linq;

    public class FemCantilever2D : FemCantileverBase
    {
        /// <summary>
        /// Number of nodes per node are 2 displacement dofs per node.
        /// </summary>
        public override int NumDofsPerNode { get => 2; }


        /// <summary>
        /// Creates a cantilever Model.
        /// </summary>
        /// <param name="numElementsPerAxis">Array with 2 entries. The number of elements on cantilever's length (0) and height (1) axis.</param>
        /// <param name="lengthPerAxis">Array with 3 entries. The entries are cantilever's length (0), height (1), width (2).</param>
        public FemCantilever2D(int[] numElementsPerAxis, double[] lengthPerAxis)
            : base(new CartesianMesh2D(numElementsPerAxis, lengthPerAxis),
                  lengthPerAxis[1] * lengthPerAxis[2] / 12 * lengthPerAxis[1] * lengthPerAxis[1]) {}

        protected override Matrix ElementStiffness()
        {
            // See "A 99 Line Topology Optimization Code Written in MATLAB", 10.1007/s001580050176
            double v = PoissonRatio;
            double E = ElasticityModulus;
            double[] k = { 0.5 - v / 6.0, 0.125 + v / 8.0, -0.25 - v / 12.0, -0.125 + 3 * v / 8.0,
                    -0.25 + v / 12.0, -0.125 - v / 8.0, v / 6.0, 0.125 - 3 * v / 8.0 }; // unique stiffness matrix entries
            var Ke = Matrix.CreateFromArray(new double[,]
            {
                { k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7] },
                    { k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2] },
                    { k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1] },
                    { k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4] },
                    { k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3] },
                    { k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6] },
                    { k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5] },
                    { k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0] }
            });
            Ke.ScaleIntoThis(E / (1 - v * v));
            return Ke;
        }

        protected override double[] CalcKnownDisplacementsForNode(double[] coords)
        {
            double x = coords[0];
            double z = coords[1] - 0.5 * Mesh.LengthPerAxis[1];
            (double u, double w) = CalcDisplacementsEulerBernoulli(x, z);
            return new double[] { u, w };
        }

        public override IGeometricMultigridModel GenerateModel(int[] numElementsPerAxis, double[] lengthPerAxis)
        { return new FemCantilever2D(numElementsPerAxis, lengthPerAxis); }
    }
}
