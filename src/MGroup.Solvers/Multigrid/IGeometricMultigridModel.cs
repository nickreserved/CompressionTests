using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization.Meshes.Structured;

namespace Compression.src.MGroup.Solvers.Multigrid
{
    public interface IGeometricMultigridModel
    {
        /// <summary>
        /// The structured mesh of the model.
        /// </summary>
        public IStructuredMesh Mesh { get; }

        /// <summary>
        /// Number of DoFs in model.
        /// </summary>
        public int NumDofsAll { get; }
        /// <summary>
        /// Number of free DoFs in model.
        /// </summary>
        public int NumDofsFree { get; }

        /// <summary>
        /// Number of nodes per node.
        /// </summary>
        public int NumDofsPerNode { get; }

        /// <summary>
        /// Generates a more simpler model.
        /// </summary>
        /// <returns>The generated simpler/coarser model.</returns>
        public IGeometricMultigridModel CreateCoarserModel();

        /// <summary>
        /// For the current model, it creates the static linear system consisted from stiffness matrix A and load vector b.
        /// </summary>
        /// <returns>First element of tuple is the stiffness matrix of model, which corresponds to free DoFs.
        /// Second element of tuple is the load vector of </returns>
        public (DokRowMajor A, Vector b) CreateLinearSystem();

        /// <summary>
        /// The IDs if the free DoFs in model. They correspond with rows and columns of the stiffness matrix.
        /// </summary>
        /// <returns>An array with free DoFs IDs.</returns>
        public int[] FindFreeDofs();

        //TODO: μελλοντικά, είναι δυνατόν, το model, να δίνει για κάθε coarser node ή DoF (array) ένα array με τους finerNodes ή finerDoFs που τον
        // επηρρεάζουν καθώς και πόσο τον επηρρεάζουν. Αυτό θα περιλαμβάνει τόσο για τους coarser όσο για για τους finer και nodes ή DoFs που θα είναι
        // παγιωμένοι. Με αυτό το array σε array, είναι πλέον εύκολο να φτιαχτούν οι restriction/interpolation πίνακες για κάθε είδους μοντέλο σε
        // επίπεδο αλγόριθμου και όχι σε επίπεδο μοντέλου. Τι γλιτώνουμε; Όχι πάρα πολλά γιατί αν έχουμε αυτό το δεδομένο, τα υπόλοιπα που μένουν
        // για να ολοκληρωθούν οι restriction/interpolation πίνακες είναι να γίνουν γραμμές/στήλες normalized στο 100% = 1 και στη συνέχεια να
        // αφαιρεθούν οι γραμμές που αφορούν fixed DoFs.

        public static void MakeCartesianCoarserElementsPerAxis(int[] numElementsPerAxis)
        {
            for (int i = 0; i < numElementsPerAxis.Length; ++i)
                if (numElementsPerAxis[i] > 1) numElementsPerAxis[i] >>= 1;
        }

        public static (IGeometricMultigridModel coarserModel, DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) CreateCoarserModelAndSmoothenerMatrices(IGeometricMultigridModel model)
        {
            IGeometricMultigridModel coarserModel = model.CreateCoarserModel();
            (DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) = CreateRestrictionAndInterpolationMatrix(model, coarserModel);
            return (coarserModel, restrictionMatrix, interpolationMatrix);
        }

        /// <summary>
        /// It generates the restriction and the interpolation matrix for current model.
        /// </summary>
        /// <param name="currentModel">The current model.</param>
        /// <param name="coarserModel">The immediately coarser model after current model. Even it can be generated from the model, to avoid the
        /// overhead of generation, it is taken as parameter (in case which already is generated). Otherwise use currentModel.CreateCoarserModel()</param>
        /// <returns>The first element of tuple is the restriction matrix which goes from current model to the coarser one.
        /// The second element of tuple is the interpolation matrix which goes from the coarser model to current model (not from the current model to
        /// finer one).</returns>
        public static (DokRowMajor restrictionMatrix, DokRowMajor interpolationMatrix) CreateRestrictionAndInterpolationMatrix(IGeometricMultigridModel currentModel, IGeometricMultigridModel coarserModel)
        {
            int[] numFinerNodesPerAxis = new int[3]; // not finerModel.Mesh.Dimension but 3
            int[] numCoarserNodesPerAxis = new int[3]; // not finerModel.Mesh.Dimension but 3
            // in every axis, how far from coarser node, there are fine nodes which influence them.
            // 2 usually (numFinerElementsPerAxis = numCoarserElementsPerAxis * 2)
            // 1 on very coarse cases - probably never (numFinerElementsPerAxis = numCoarserElementsPerAxis = 1)
            // 2-2.5 on other cases (numFinerElementsPerAxis > numCoarserElementsPerAxis * 2
            //                   but numFinerElementsPerAxis / 2 = numCoarserElementsPerAxis)
            double[] nodeInfluenceDistance = new double[currentModel.Mesh.Dimension];
            {
                int[] numFinerElementsPerAxis = ((ICartesianMesh)currentModel.Mesh).NumElements;   //ref
                int[] numCoarserElementsPerAxis = ((ICartesianMesh)coarserModel.Mesh).NumElements; //ref;
                // for everyone of 2 or 3 axis, we calculate influence window size from finer Model to coarser.
                for (int i = 0; i < currentModel.Mesh.Dimension; ++i)
                {
                    nodeInfluenceDistance[i] = (double)numFinerElementsPerAxis[i] / numCoarserElementsPerAxis[i];

                    numFinerNodesPerAxis[i] = numFinerElementsPerAxis[i] + 1;
                    numCoarserNodesPerAxis[i] = numCoarserElementsPerAxis[i] + 1;
                }
            }
            if (currentModel.Mesh.Dimension < 3) numFinerNodesPerAxis[2] = numCoarserNodesPerAxis[2] = 1;

            DokRowMajor restriction = DokRowMajor.CreateEmpty(coarserModel.NumDofsAll, currentModel.NumDofsAll);
            DokRowMajor interpolation = DokRowMajor.CreateEmpty(currentModel.NumDofsAll, coarserModel.NumDofsAll);

            int[] a = new int[3];            // coordinates of coarser Model's current node, in coarser Model's Mesh coordinate system
            double[] cursor = new double[3]; // coordinates of coarser Model's current node, in finer Model's Mesh coordinate system
            int[] start = new int[3];        // influence window's start dimensions in finer Model's Mesh coordinate system (for coarser Model's current node)
            int[] end = new int[3];          // influence window's end dimensions in finer Model's Mesh coordinate system (for coarser Model's current node)
            int[] w = new int[3];            // coordinates of finer Model's current node in influence window, in finer Model's Mesh coordinate system
            // sum for each row of interpolation matrix. Because restriction matrix created row by row, that means that interpolation
            // matrix created column by column but it is row major. So sums of Rows are vector and it is applied on the end of processing
            double[] sumInterpolationRow = new double[currentModel.Mesh.NumNodesTotal];
            if (currentModel.Mesh.Dimension < 3) end[2] = 1;// avoid calculation and also avoid premature end of for()
            // 3 for, for every coarser Model's node
            for (a[0] = 0; a[0] < numCoarserNodesPerAxis[0]; ++a[0])
            {
                for (a[2] = 0; a[2] < numCoarserNodesPerAxis[2]; ++a[2])
                {
                    for (a[1] = 0; a[1] < numCoarserNodesPerAxis[1]; ++a[1])
                    {
                        // calculate influence window for coarser Model's current node
                        // clipping in boundaries
                        for (int i = 0; i < currentModel.Mesh.Dimension; ++i)
                        {
                            cursor[i] = a[i] * nodeInfluenceDistance[i];
                            start[i] = (int)Math.Ceiling(Math.BitIncrement(cursor[i] - nodeInfluenceDistance[i])); // from including
                            if (start[i] < 0) start[i] = 0;
                            end[i] = (int)Math.Floor(Math.BitDecrement(cursor[i] + nodeInfluenceDistance[i])) + 1;    // to excluding
                            if (end[i] > numFinerNodesPerAxis[i]) end[i] = numFinerNodesPerAxis[i];
                        }
                        // the id of the coarser Model's node. multiplied by dofsPerNode gives the row of restriction matrix
                        int coarserNodeId = coarserModel.Mesh.GetNodeID(a[..currentModel.Mesh.Dimension]);
                        int coarserDofBaseId = coarserNodeId * coarserModel.NumDofsPerNode;
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
                                    int finerNodeId = currentModel.Mesh.GetNodeID(w[..currentModel.Mesh.Dimension]);
                                    int finerDofBaseId = finerNodeId * currentModel.NumDofsPerNode;

                                    double nodeInfluence = 0;
                                    for (int i = 0; i < currentModel.Mesh.Dimension; ++i)
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
                            for (int i = 1; i < currentModel.NumDofsPerNode; ++i)
                                restriction[coarserDofBaseId + i, key + i] = value;
                        }
                    }
                }
            }
            // now we can normalize the interpolation matrix. Every column is in place. Also DoFs other than 0 populated.
            for (int finerNodeId = 0; finerNodeId < currentModel.Mesh.NumNodesTotal; ++finerNodeId)
            {
                int finerDofBaseId = finerNodeId * currentModel.NumDofsPerNode;
                Dictionary<int, double> interpolationRow = interpolation.RawRows[finerDofBaseId];
                foreach (var key in interpolationRow.Keys.ToList())
                {
                    double value = interpolationRow[key] / sumInterpolationRow[finerNodeId];
                    interpolationRow[key] = value;
                    for (int i = 1; i < currentModel.NumDofsPerNode; ++i)
                        interpolation[finerDofBaseId + i, key + i] = value;
                }
            }
            // After normalization takes place we must return only a submatrix for both restriction and interpolation with only free DoFs.
            // This must be happen after normalization because fixed DoFs influence normalization. Of course after that, not every row has sum of 1 = 100%.
            int[] finerFreeDofs = currentModel.FindFreeDofs();
            int[] coarserFreeDofs = coarserModel.FindFreeDofs();
            restriction = restriction.GetSubmatrix(coarserFreeDofs, finerFreeDofs);
            interpolation = interpolation.GetSubmatrix(finerFreeDofs, coarserFreeDofs);

            return (restriction, interpolation);
        }
    }
}
