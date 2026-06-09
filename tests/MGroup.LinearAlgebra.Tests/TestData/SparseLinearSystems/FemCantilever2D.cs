namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    using global::MGroup.LinearAlgebra.Matrices;
    using global::MGroup.MSolve.Discretization.Meshes.Structured;

    public class FemCantilever2D : FemCantileverBase
    {
        /// <summary>
        /// Number of nodes per node are 2 displacement dofs per node.
        /// </summary>
        public override int NumDofsPerNode { get => 2; }
        /// <summary>
        /// Depth of cantilever.
        /// </summary>
        public readonly double CantileverDepth;

        /// <summary>
        /// Creates a cantilever Model.
        /// </summary>
        /// <param name="numElementsPerAxis">Array with 2 entries. The number of elements on cantilever's length (0) and height (1) axis.</param>
        /// <param name="lengthPerAxis">Array with 3 entries. The entries are cantilever's length (0), height (1), width (2).</param>
        public FemCantilever2D(int[] numElementsPerAxis, double[] lengthPerAxis)
            : base(new UniformCartesianMesh2D.Builder(new double[] { 0, 0 }, lengthPerAxis[..2], numElementsPerAxis).BuildMesh(),
                  lengthPerAxis[1] * lengthPerAxis[2] / 12 * lengthPerAxis[1] * lengthPerAxis[1])
        {
            CantileverDepth = lengthPerAxis[2];
        }

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
            double z = coords[1] - (Mesh.MaxCoordinates[1] - Mesh.MinCoordinates[1]) / 2;
            (double u, double w) = CalcDisplacementsEulerBernoulli(x, z);
            return new double[] { u, w };
        }

        protected override FemCantilever2D GenerateModel(int[] numElementsPerAxis, double[] lengthPerAxis)
        { return new FemCantilever2D(numElementsPerAxis, lengthPerAxis.Append(CantileverDepth).ToArray()); }
    }
}
