using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Structural.BoundaryConditions;
using MGroup.FEM.Structural.Tests.Plates.FemExtentions;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.DataStructures;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Entities;

namespace MGroup.FEM.Structural.Tests.Plates.Commons
{
	public class BodyForcesConverter
	{
		public static List<NodalLoad> ApplyUniformLoadOnPlateElement(PlateElementRectangle4Nodes3Dofs element, double uniformDistributedLoad)
		{
			var result = new List<NodalLoad>();
			Vector elementForces = element.ConvertBodyForcesToNodal(uniformDistributedLoad);
			for (int n = 0; n < element.Nodes.Count; n++)
			{
				INode node = element.Nodes[n];
				result.Add(new NodalLoad(node, StructuralDof.TranslationZ, elementForces[3 * n]));
				result.Add(new NodalLoad(node, StructuralDof.RotationX, elementForces[3 * n + 1]));
				result.Add(new NodalLoad(node, StructuralDof.RotationY, elementForces[3 * n + 2]));
			}
			return result;
		}

		public static List<NodalLoad> ApplyUniformLoadOnAllPlateElements(
			Model model, double uniformDistributedLoad, List<NodalDisplacement> dirichletBCs)
		{
			// Find supported dofs
			var supportedDofs = new Dictionary<int, HashSet<IDofType>>();
			foreach (NodalDisplacement bc in dirichletBCs)
			{
				bool nodeExists = supportedDofs.TryGetValue(bc.Node.ID, out HashSet<IDofType> dofs);
				if (!nodeExists)
				{
					dofs = new HashSet<IDofType>();
					supportedDofs[bc.Node.ID] = dofs;
				}

				dofs.Add(bc.DOF);
			}

			// Gather the nodal loads from each element
			var allNodalLoads = new Table<INode, IStructuralDofType, double>();
			foreach (IElementType element in model.ElementsDictionary.Values)
			{
				var plateElement = (PlateElementRectangle4Nodes3Dofs)element;
				List<NodalLoad> equivalentForces = ApplyUniformLoadOnPlateElement(plateElement, uniformDistributedLoad);
				foreach (NodalLoad load in equivalentForces)
				{
					bool exists = allNodalLoads.TryGetValue(load.Node, load.DOF, out double currentVal);
					double sum = exists ? currentVal + load.Amount : load.Amount;
					allNodalLoads[load.Node, load.DOF] = sum;
				}
			}

			// Return them as List<NodalLoad>
			var result = new List<NodalLoad>();
			foreach ((INode node, IStructuralDofType dof, double val) in allNodalLoads)
			{
				// Skip this dof if it is supported
				if (supportedDofs.TryGetValue(node.ID, out HashSet<IDofType> dofs))
				{
					if (dofs.Contains(dof))
					{
						continue;
					}
				}
				result.Add(new NodalLoad(node, dof, val));
			}
			return result;
		}
	}
}
