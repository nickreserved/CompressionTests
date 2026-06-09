namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    using Compression.src.MGroup.Solvers.Multigrid;
    using global::MGroup.LinearAlgebra.Matrices;
    using global::MGroup.LinearAlgebra.Matrices.Builders;
    using global::MGroup.LinearAlgebra.Vectors;
    using global::MGroup.MSolve.Discretization.Meshes.Structured;
    using System;
    using System.Linq;

    abstract public class FemCantileverBase : IGeometricMultigridModel
    {
        public ICartesianMesh Mesh { get; }
        IStructuredMesh IGeometricMultigridModel.Mesh { get => Mesh; }
        public int NumDofsAll { get; private set; }
        public int NumDofsFree { get; private set; }

        public double[] ElementDensities { get; set; } = null;
        public double ElasticityModulus { get; set; } = 1.0;
        public double PoissonRatio { get; set; } = 0.3;
        public double DistributedLoad { get; set; } = 1.0;
        public double MomentOfInertia { get; }

        /// <summary>
        /// Number of nodes per node.
        /// </summary>
        /// <remarks>
        /// 1d mesh has 2 displacement and one rotational dofs per node.<br/>
        /// 2d mesh has 2 displacement dofs per node.<br/>
        /// 3d mesh has 3 displacement dofs per node.
        /// </remarks>
        abstract public int NumDofsPerNode { get; }


        /// <summary>
        /// Creates a cantilever Model.
        /// </summary>
        /// <param name="numElementsPerAxis">Array with 1, 2 or 3 entries.<br/>
        /// 1 for 1d mesh, on length axis of cantilever,<br/>
        /// 2 for 2d mesh, on length (0) and height (1) axis of cantilever,<br/>
        /// 3 for 3d mesh, on length (0), width (1) and height (2) axis of cantilever.</param>
        /// <param name="lengthPerAxis">Array with 3 entries.<br/>
        /// For 1d mesh and 2d mesh the entries are cantilever's length (0), height (1), width (2)<br/>
        /// For 3d mesh the entries are cantilever's length (0), width (1), height (2).</param>
        /// <exception cref="ArgumentException">If <paramref name="lengthPerAxis"/> has not 3 entries</exception>
        public FemCantileverBase(ICartesianMesh mesh, double momentOfInertia)
        {
            Mesh = mesh;
            MomentOfInertia = momentOfInertia;
            NumDofsAll = Mesh.Dimension * Mesh.NumNodesTotal;
            NumDofsFree = NumDofsAll - NumDofsAll / Mesh.NumNodes[0]; // nodes with x = 0 are constrained
        }
        public (DokRowMajor A, Vector x, Vector b) CreateAndSolveLinearSystem()
        {
            UpdateElementDensities();
            DokRowMajor K = AssembleGlobalMatrix();
            int[] freeDofs = FindFreeDofs();
            DokRowMajor Kff = K.GetSubmatrix(freeDofs, freeDofs);
            Vector Uf = CalcFreeDisplacements();
            Vector Ff = Kff.MultiplyRight(Uf);
            return (Kff, Uf, Ff);
        }

        public (DokRowMajor A, Vector b) CreateLinearSystem()
        {
            (DokRowMajor A, _, Vector b) = CreateAndSolveLinearSystem();
            return (A, b);
        }

        protected (double u, double w) CalcDisplacementsEulerBernoulli(double x, double z)
        {
            double L = Mesh.MaxCoordinates[0] - Mesh.MinCoordinates[0];
            double q = DistributedLoad;
            double E = ElasticityModulus;
            double I = MomentOfInertia;

            double w = -q / (24 * E * I) * (Math.Pow(x, 4) - 4 * L * Math.Pow(x, 3) + 6 * L * L * x * x);
            double rot = -q / (6 * E * I) * (Math.Pow(x, 3) - 3 * L * x * x + 3 * L * L * x);
            double u = -rot * z;

            return (u, w);
        }

        abstract protected Matrix ElementStiffness();
        abstract protected double[] CalcKnownDisplacementsForNode(double[] coords);

        private DokRowMajor AssembleGlobalMatrix()
        {
            Matrix Ke = ElementStiffness();
            int numElementDofs = Ke.NumRows;
            int[] elementDofsLocal = Enumerable.Range(0, numElementDofs).ToArray();

            int numAllDofs = NumDofsAll;
            DokRowMajor Kglob = DokRowMajor.CreateEmpty(numAllDofs, numAllDofs);
            for (int e = 0; e < Mesh.NumElementsTotal; e++)
            {
                int[] elementIdx = Mesh.GetElementIdx(e);
                int[] elementDofsGlobal = GetElementDofsGlobal(elementIdx);
                double density = ElementDensities[e];
                Kglob.AddSubmatrix(density * Ke, elementDofsLocal, elementDofsGlobal, elementDofsLocal, elementDofsGlobal);
            }

            return Kglob;
        }

        /// <summary>
        /// The global indices of the element's dofs, in 0-based numbering.
        /// </summary>
        private int[] GetElementDofsGlobal(int[] elementIdx)
        {
            int[] nodeIds = Mesh.GetElementConnectivity(elementIdx);
            var globalDofs = new int[NumDofsPerNode * nodeIds.Length];
            for (int i = 0; i < nodeIds.Length; ++i)
            {
                // 2D: 2*n, 2*n+1. 3D: 3*n, 3*n+1, 3*n+2
                for (int d = 0; d < NumDofsPerNode; ++d)
                {
                    globalDofs[NumDofsPerNode * i + d] = NumDofsPerNode * nodeIds[i] + d;
                }
            }

            return globalDofs;
        }

        private void UpdateElementDensities()
        {
            if (ElementDensities == null)
            {
                ElementDensities = new double[Mesh.NumElementsTotal];
                for (int e = 0; e < Mesh.NumElementsTotal; e++)
                {
                    ElementDensities[e] = 1.0;
                }
            }
        }

        public int[] FindFreeDofs()
        {
            int dofsPerNode = NumDofsPerNode;
            var freeDofs = new int[NumDofsFree];
            int pos = -1;
            for (int n = 0; n < Mesh.NumNodesTotal; ++n)
            {
                int[] nodeIdx = Mesh.GetNodeIdx(n);
                if (nodeIdx[0] == 0)
                {
                    continue; // constrained nodes at x=0 (finerNodeId=0);
                }

                // 2D: 2*n, 2*n+1. 3D: 3*n, 3*n+1, 3*n+2
                for (int d = 0; d < dofsPerNode; ++d)
                {
                    freeDofs[++pos] = dofsPerNode * n + d;
                }
            }

            return freeDofs;
        }

        private Vector CalcFreeDisplacements()
        {
            var Uf = new double[NumDofsFree];
            int pos = -1;
            for (int n = 0; n < Mesh.NumNodesTotal; ++n)
            {
                int[] nodeIdx = Mesh.GetNodeIdx(n);
                if (nodeIdx[0] == 0)
                {
                    continue; // constrained nodes at x=0 (finerNodeId=0);
                }

                double[] coords = Mesh.GetNodeCoordinates(nodeIdx);
                double[] displ = CalcKnownDisplacementsForNode(coords);

                for (int i = 0; i < displ.Length; ++i)
                {
                    Uf[++pos] = displ[i];
                }
            }

            return Vector.CreateFromArray(Uf);
        }

        abstract protected FemCantileverBase GenerateModel(int[] numElementsPerAxis, double[] lengthPerAxis);

        public IGeometricMultigridModel CreateCoarserModel()
        {
            int[] numElementsPerAxis = (int[])Mesh.NumElements.Clone();
            IGeometricMultigridModel.MakeCartesianCoarserElementsPerAxis(numElementsPerAxis);
            return GenerateModel(numElementsPerAxis, Mesh.MaxCoordinates.Zip(Mesh.MinCoordinates, (x, y) => x - y).ToArray());
        }
    }
}
