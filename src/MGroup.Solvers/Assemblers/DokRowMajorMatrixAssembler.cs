using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Xml.Linq;

using MGroup.LinearAlgebra.Matrices;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Providers;
using MGroup.Solvers.DofOrdering;
using MGroup.Solvers.LinearAlgebraExtensions.Matrices.Builders;

namespace MGroup.Solvers.Assemblers
{
	public class DokRowMajorMatrixAssembler : ISubdomainMatrixAssembler<DokRowMajor>
	{
		private const string name = "DokRowMajorMatrixAssembler"; // for error messages
		private readonly bool isSymmetric;
		//private ConstrainedMatricesAssembler constrainedAssembler = new ConstrainedMatricesAssembler();

		bool isIndexerCached = false;
		private int[] cachedColIndices, cachedRowOffsets;

		public DokRowMajorMatrixAssembler(bool isSymmetric = true)
		{
			this.isSymmetric = isSymmetric;
		}

		public DokRowMajor BuildGlobalMatrix(ISubdomainFreeDofOrdering dofOrdering, IEnumerable<IElementType> elements,
			IElementMatrixProvider matrixProvider)
		{
			int numFreeDofs = dofOrdering.NumFreeDofs;
			var subdomainMatrix = DokRowMajor.CreateEmpty(numFreeDofs, numFreeDofs);

			IReadOnlyMatrix elementMatrix = null;

			foreach (IElementType element in elements)
			{
				if (elementMatrix == null)
				{
					elementMatrix = matrixProvider.Matrix(element);
				}

				(int[] elementDofIndices, int[] subdomainDofIndices) = dofOrdering.MapFreeDofsElementToSubdomain(element);
				if (isSymmetric)
				{
					subdomainMatrix.AddSubmatrixSymmetric(elementMatrix, elementDofIndices, subdomainDofIndices);
				}
				else
				{
					subdomainMatrix.AddSubmatrix(elementMatrix, elementDofIndices, subdomainDofIndices, 
						elementDofIndices, subdomainDofIndices);
				}
			}
			
			return subdomainMatrix;
		}

		public DokRowMajorMatrixAssembler Clone() => new DokRowMajorMatrixAssembler(isSymmetric);

        public DokRowMajor CreateEmptyMatrix(ISubdomainFreeDofOrdering dofOrdering)
        {
            throw new NotImplementedException();
        }

        public void HandleDofOrderingWasModified()
		{
			//TODO: perhaps the indexer should be disposed altogether. Then again it could be in use by other matrices.
			cachedColIndices = null;
			cachedRowOffsets = null;
			isIndexerCached = false;
		}

        ISubdomainMatrixAssembler<DokRowMajor> ISubdomainMatrixAssembler<DokRowMajor>.Clone()
        {
            return Clone();
        }
    }
}
