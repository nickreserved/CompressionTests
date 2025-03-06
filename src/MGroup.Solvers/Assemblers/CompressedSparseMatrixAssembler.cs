using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Matrices.Builders;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Providers;
using MGroup.Solvers.DofOrdering;
using System.Diagnostics;

//TODO: Instead of storing the raw CSR arrays, use a reusable DOK or CsrIndexer class. That class should provide methods to 
//      assemble the values part of the global matrix more efficiently than the general purpose DOK. The general purpose DOK 
//      should only be used to assemble the first global matrix and whenever the dof ordering changes. Now it is used everytime 
//      and the indexing arrays are discarded.
namespace MGroup.Solvers.Assemblers
{
    /// <summary>
    /// Builds the global matrix of the linear system that will be solved. This matrix is square and stored in CSR format, but
    /// both triangles are explicitly stored. This format is suitable for matrix/vector multiplications, therefore it can be 
    /// combined with many iterative solvers. 
    /// Authors: Serafeim Bakalakos
    /// </summary>
    public class CompressedSparseMatrixAssembler : ISubdomainMatrixAssembler<DuViCompressedSparseMatrix>
    {
        private const string name = "CompressedSparseAssembler"; // for error messages
        //private ConstrainedMatricesAssembler constrainedAssembler = new ConstrainedMatricesAssembler();

        bool isIndexerCached = false;
        private int[] cachedColIndices, cachedRowOffsets;

		public DuViCompressedSparseMatrix CreateEmptyMatrix(ISubdomainFreeDofOrdering dofOrdering) =>
            new DuViCompressedSparseMatrix(dofOrdering.NumFreeDofs, dofOrdering.NumFreeDofs);

        public DuViCompressedSparseMatrix BuildGlobalMatrix(ISubdomainFreeDofOrdering dofOrdering, IEnumerable<MGroup.MSolve.Discretization.IElementType> elements, 
            IElementMatrixProvider matrixProvider)
        {
            int numFreeDofs = dofOrdering.NumFreeDofs;
            var subdomainMatrix = DokRowMajor.CreateEmpty(numFreeDofs, numFreeDofs);

            foreach (IElementType element in elements)
            {
                (int[] elementDofIndices, int[] subdomainDofIndices) = dofOrdering.MapFreeDofsElementToSubdomain(element);
                IMatrix elementMatrix = matrixProvider.Matrix(element);
                subdomainMatrix.AddSubmatrixSymmetric(elementMatrix, elementDofIndices, subdomainDofIndices);
            }

            (double[] values, int[] colIndices, int[] rowOffsets) = subdomainMatrix.BuildCsrArrays(true);
            if (!isIndexerCached)
            {
                cachedColIndices = colIndices;
                cachedRowOffsets = rowOffsets;
                isIndexerCached = true;
            }
            else
            {
                Debug.Assert(Utilities.AreEqual(cachedColIndices, colIndices));
                Debug.Assert(Utilities.AreEqual(cachedRowOffsets, rowOffsets));
            }
            return new DuViCompressedSparseMatrix(numFreeDofs, numFreeDofs, values, cachedColIndices, cachedRowOffsets);
        }

		public ISubdomainMatrixAssembler<DuViCompressedSparseMatrix> Clone() => new CompressedSparseMatrixAssembler();

		public void HandleDofOrderingWasModified()
        {
            //TODO: perhaps the indexer should be disposed altogether. Then again it could be in use by other matrices.
            cachedColIndices = null;
            cachedRowOffsets = null;
            isIndexerCached = false;
        }
    }
}
