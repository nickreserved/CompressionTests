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
        IStructuredMesh IStructuredModel.Mesh { get => Mesh; }
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

        abstract public IGeometricMultigridModel GenerateModel(int[] numElementsPerAxis, double[] lengthPerAxis);

        public IStructuredModel GenerateModel(int detail)
        {
            int[] numElementsPerAxis = (int[])Mesh.NumElements.Clone();
            if (detail == 0) throw new ArgumentException("parameter 'detail' must not be 0");
            if (detail > 0)
                for (int i = 0; i < numElementsPerAxis.Length; ++i)
                    numElementsPerAxis[i] <<= detail;
            else
            {
                int threshold = 1 << -detail;
                for (int i = 0; i < numElementsPerAxis.Length; ++i)
                    if (numElementsPerAxis[i] > threshold) numElementsPerAxis[i] >>= -detail;
                    else numElementsPerAxis[i] = 1;
            }
            return GenerateModel(numElementsPerAxis, Mesh.MaxCoordinates.Zip(Mesh.MinCoordinates, (x, y) => x - y).ToArray());
        }

        public (DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) CreateRestrictionAndInterpolationMatrix(IStructuredModel coarserModel)
        {
            int[] numFinerNodesPerAxis = new int[3]; // not finerModel.Mesh.Dimension but 3
            int[] numCoarserNodesPerAxis = new int[3]; // not finerModel.Mesh.Dimension but 3
            //int dofsPerNode = finerModel.NumDofsPerNode;
            // in every axis, how far from coarser node, there are fine nodes which influence them.
            // 2 usually (numFinerElementsPerAxis = numCoarserElementsPerAxis * 2)
            // 1 on very coarse cases - probably never (numFinerElementsPerAxis = numCoarserElementsPerAxis = 1)
            // 2-2.5 on other cases (numFinerElementsPerAxis > numCoarserElementsPerAxis * 2
            //                   but numFinerElementsPerAxis / 2 = numCoarserElementsPerAxis)
            double[] nodeInfluenceDistance = new double[3]; // not dofsPerNode but 3
            // The 1d/2d/3d table of influence of current (fine) nodes in one coarse node, is always the same if rollingWindow is false.
            // for instance for 1d table can be for one coarse node: 1 2 1.
            // If rollingWindow is true coarse node influenced from a non-integer number of current (fine) nodes per direction. So the table of
            // influence is changing for every coarse node.
            bool rollingWindow = true; // can optimize for speed if false, but code for optimization didn't written.
            {
                int[] numFinerElementsPerAxis = Mesh.NumElements;   //ref
                int[] numCoarserElementsPerAxis = ((FemCantileverBase)coarserModel).Mesh.NumElements; //ref;
                // for everyone of 2 or 3 axis, we calculate influence window size from finer Model to coarser.
                for (int i = 0; i < Mesh.Dimension; ++i)
                {
                    nodeInfluenceDistance[i] = (double)numFinerElementsPerAxis[i] / numCoarserElementsPerAxis[i];

                    numFinerNodesPerAxis[i] = numFinerElementsPerAxis[i] + 1;
                    numCoarserNodesPerAxis[i] = numCoarserElementsPerAxis[i] + 1;

                    rollingWindow &= Math.Truncate(nodeInfluenceDistance[i]) == nodeInfluenceDistance[i];
                }
            }
            if (Mesh.Dimension < 3) numFinerNodesPerAxis[2] = numCoarserNodesPerAxis[2] = 1;

            DokRowMajor restriction = DokRowMajor.CreateEmpty(coarserModel.NumDofsAll, NumDofsAll);
            DokRowMajor interpolation = DokRowMajor.CreateEmpty(NumDofsAll, coarserModel.NumDofsAll);

            int[] a = new int[3];            // coordinates of coarser Model's current node, in coarser Model's Mesh coordinate system
            double[] cursor = new double[3]; // coordinates of coarser Model's current node, in finer Model's Mesh coordinate system
            int[] start = new int[3];        // influence window's start dimensions in finer Model's Mesh coordinate system (for coarser Model's current node)
            int[] end = new int[3];          // influence window's end dimensions in finer Model's Mesh coordinate system (for coarser Model's current node)
            int[] w = new int[3];            // coordinates of finer Model's current node in influence window, in finer Model's Mesh coordinate system
            // sum for each row of interpolation matrix. Because restriction matrix created row by row, that means that interpolation
            // matrix created column by column but it is row major. So sums of Rows are vector and it is applied on the end of processing
            double[] sumInterpolationRow = new double[Mesh.NumNodesTotal];
            if (Mesh.Dimension < 3) end[2] = 1;// avoid calculation and also avoid premature end of for()
            // 3 for, for every coarser Model's node
            for (a[0] = 0; a[0] < numCoarserNodesPerAxis[0]; ++a[0])
            {
                for (a[2] = 0; a[2] < numCoarserNodesPerAxis[2]; ++a[2])
                {
                    for (a[1] = 0; a[1] < numCoarserNodesPerAxis[1]; ++a[1])
                    {
                        // calculate influence window for coarser Model's current node
                        // clipping in boundaries
                        for (int i = 0; i < Mesh.Dimension; ++i)
                        {
                            cursor[i] = a[i] * nodeInfluenceDistance[i];
                            start[i] = (int)Math.Ceiling(Math.BitIncrement(cursor[i] - nodeInfluenceDistance[i])); // from including
                            if (start[i] < 0) start[i] = 0;
                            end[i] = (int)Math.Floor(Math.BitDecrement(cursor[i] + nodeInfluenceDistance[i])) + 1;    // to excluding
                            if (end[i] > numFinerNodesPerAxis[i]) end[i] = numFinerNodesPerAxis[i];
                        }
                        // the id of the coarser Model's node. multiplied by dofsPerNode gives the row of restriction matrix
                        int coarserNodeId = coarserModel.Mesh.GetNodeID(a[..Mesh.Dimension]);
                        int coarserDofBaseId = coarserNodeId * ((FemCantileverBase)coarserModel).NumDofsPerNode;
                        // sum for each row of restriction matrix. Because restriction matrix created row by row, it is scalar and it is applied on
                        // the end of each row
                        double sumRestrictionRow = 0;
                        // populate 2 matrices with influences but not normalized yet (normalization means: every row must have sum of 1 = 100%)
                        // process finer Model's nodes inside the influence window for coarser Model's current node
                        for (w[0] = start[0]; w[0] < end[0]; ++w[0])
                            for (w[2] = start[2]; w[2] < end[2]; ++w[2])
                                for (w[1] = start[1]; w[1] < end[1]; ++w[1])
                                {
                                    // the id of the finer Model's node. multiplied by dofsPerNode gives the column of restriction matrix
                                    int finerNodeId = Mesh.GetNodeID(w[..Mesh.Dimension]);
                                    int finerDofBaseId = finerNodeId * NumDofsPerNode;

                                    double nodeInfluence = 0;
                                    for (int i = 0; i < Mesh.Dimension; ++i)
                                    {
                                        double d0 = (w[i] - cursor[i]) / nodeInfluenceDistance[i];
                                        nodeInfluence += d0 * d0;
                                    }
                                    nodeInfluence = 1 - Math.Sqrt(nodeInfluence);
                                    sumRestrictionRow += nodeInfluence;
                                    sumInterpolationRow[finerNodeId] += nodeInfluence;

                                    // population of restriction matrix and interpolation matrix, only for DoF 0. Where is the other DoFs?
                                    // They will be populated with the same values after the normalization.
                                    restriction[coarserDofBaseId, finerDofBaseId] = nodeInfluence;
                                    interpolation[finerDofBaseId, coarserDofBaseId] = nodeInfluence;
                                }
                        // normalization of restriction matrix current row. Also DoFs other than 0 populated.
                        // This cannot be happen here for interpolation matrix, because next columns are not placed yet, so Rows are not complete yet.
                        Dictionary<int, double> restrictionRow = restriction.RawRows[coarserDofBaseId];
                        foreach (var key in restrictionRow.Keys.ToList())
                        {
                            double value = restrictionRow[key] / sumRestrictionRow;
                            restrictionRow[key] = value;
                            for (int i = 1; i < NumDofsPerNode; ++i)
                                restriction[coarserDofBaseId + i, key + i] = value;
                        }
                    }
                }
            }
            // now we can normalize the interpolation matrix. Every column is in place. Also DoFs other than 0 populated.
            for (int finerNodeId = 0; finerNodeId < Mesh.NumNodesTotal; ++finerNodeId)
            {
                int finerDofBaseId = finerNodeId * NumDofsPerNode;
                Dictionary<int, double> interpolationRow = interpolation.RawRows[finerDofBaseId];
                foreach (var key in interpolationRow.Keys.ToList())
                {
                    double value = interpolationRow[key] / sumInterpolationRow[finerNodeId];
                    interpolationRow[key] = value;
                    for (int i = 1; i < NumDofsPerNode; ++i)
                        interpolation[finerDofBaseId + i, key + i] = value;
                }
            }
            // After normalization takes place we must return only a submatrix for both restriction and interpolation with only free DoFs.
            // This must be happen after normalization because fixed DoFs influence normalization. Of course after that, not every row has sum of 1 = 100%.
            int[] finerFreeDofs = FindFreeDofs();
            int[] coarserFreeDofs = coarserModel.FindFreeDofs();
            restriction = restriction.GetSubmatrix(coarserFreeDofs, finerFreeDofs);
            interpolation = interpolation.GetSubmatrix(finerFreeDofs, coarserFreeDofs);

            return (restriction, interpolation);
        }

        public (IGeometricMultigridModel coarserModel, DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) CreateCoarserModelAndSmoothenerMatrices()
        {
            IGeometricMultigridModel coarserModel = CreateCoarserModel();
            (DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) = CreateRestrictionAndInterpolationMatrix(coarserModel);
            return (coarserModel, restrictionMatrix, interpolationMatrix);
        }

        public IGeometricMultigridModel CreateCoarserModel()
        {
            int[] numElementsPerAxis = (int[]) Mesh.NumElements.Clone();
            for (int i = 0; i < numElementsPerAxis.Length; ++i)
                if (numElementsPerAxis[i] > 1) numElementsPerAxis[i] >>= 1;
            return GenerateModel(numElementsPerAxis, Mesh.MaxCoordinates.Zip(Mesh.MinCoordinates, (x, y) => x - y).ToArray());
        }

        public bool IsDofFree(int dof)
        {
            throw new NotImplementedException();
        }
    }
}
