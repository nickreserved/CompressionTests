using MGroup.FEM.Structural.Tests.Plates.FemExtentions;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization;
using MGroup.Solvers.DofOrdering;

namespace MGroup.Solvers.Assemblers
{
	public class PlateElementVectorAssembler
	{



		public Vector BuildGlobalVector(ISubdomainFreeDofOrdering dofOrdering, IEnumerable<IElementType> elements, double loadZ, bool sameLoadAndElements)
		{
			int numFreeDofs = dofOrdering.NumFreeDofs;
			var subdomainVector = Vector.CreateZero(numFreeDofs);

			if (sameLoadAndElements)
			{
				Vector elementVector = null;
				foreach (IElementType element in elements)
				{
					if (elementVector == null)
					{
						var plateElement = (PlateElementRectangle4Nodes3Dofs)element;
						elementVector = plateElement.ConvertBodyForcesToNodal(loadZ);
					}
					
					(int[] elementDofIndices, int[] subdomainDofIndices) = dofOrdering.MapFreeDofsElementToSubdomain(element);
					subdomainVector.AddIntoThisNonContiguouslyFrom(subdomainDofIndices, elementVector, elementDofIndices);
				}
			}
			else
			{
				foreach (IElementType element in elements)
				{
					var plateElement = (PlateElementRectangle4Nodes3Dofs)element;
					Vector elementVector = plateElement.ConvertBodyForcesToNodal(loadZ);

					(int[] elementDofIndices, int[] subdomainDofIndices) = dofOrdering.MapFreeDofsElementToSubdomain(element);
					subdomainVector.AddIntoThisNonContiguouslyFrom(subdomainDofIndices, elementVector, elementDofIndices);
				}
			}

			return subdomainVector;
		}
	}
}
