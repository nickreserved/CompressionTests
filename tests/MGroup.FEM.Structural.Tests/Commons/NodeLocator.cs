using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MGroup.MSolve.Discretization.Entities;

namespace MGroup.FEM.Structural.Tests.Plates.Commons
{
	public class NodeLocator
	{
		private Model model;
		private readonly double tolerance;

		public NodeLocator(Model model, double distanceTolerance)
		{
			this.model = model;
			tolerance = distanceTolerance;
		}

		public INode FindNodeWithXY(double targetX, double targetY, double targetZ)
		{
			foreach (INode node in model.NodesDictionary.Values)
			{
				if (Math.Abs(node.X - targetX) < tolerance && Math.Abs(node.Y - targetY) < tolerance
					&& Math.Abs(node.Z - targetZ) < tolerance)
				{
					return node;
				}
			}
			return null;
		}

		public List<INode> FindNodesWithX(double targetX)
		{
			var result = new List<INode>();
			foreach (INode node in model.NodesDictionary.Values)
			{
				if (Math.Abs(node.X - targetX) < tolerance)
				{
					result.Add(node);
				}
			}
			return result;
		}

		public List<INode> FindNodesWithXY(double targetX, double targetY)
		{
			var result = new List<INode>();
			foreach (INode node in model.NodesDictionary.Values)
			{
				if (Math.Abs(node.X - targetX) < tolerance && Math.Abs(node.Y - targetY) < tolerance)
				{
					result.Add(node);
				}
			}
			return result;
		}

		public List<INode> FindNodesWithXZ(double targetX, double targetZ)
		{
			var result = new List<INode>();
			foreach (INode node in model.NodesDictionary.Values)
			{
				if (Math.Abs(node.X - targetX) < tolerance && Math.Abs(node.Z - targetZ) < tolerance)
				{
					result.Add(node);
				}
			}
			return result;
		}

		public List<INode> FindNodesWithY(double targetY)
		{
			var result = new List<INode>();
			foreach (INode node in model.NodesDictionary.Values)
			{
				if (Math.Abs(node.Y - targetY) < tolerance)
				{
					result.Add(node);
				}
			}
			return result;
		}

		public List<INode> FindNodesWithYZ(double targetY, double targetZ)
		{
			var result = new List<INode>();
			foreach (INode node in model.NodesDictionary.Values)
			{
				if (Math.Abs(node.Y - targetY) < tolerance && Math.Abs(node.Z - targetZ) < tolerance)
				{
					result.Add(node);
				}
			}
			return result;
		}

		public List<INode> FindNodesWithZ(double targetZ)
		{
			var result = new List<INode>();
			foreach (INode node in model.NodesDictionary.Values)
			{
				if (Math.Abs(node.Z - targetZ) < tolerance)
				{
					result.Add(node);
				}
			}
			return result;
		}

		public List<INode> FindNodesWith(Predicate<INode> filter)
		{
			var result = new List<INode>();
			foreach (INode node in model.NodesDictionary.Values)
			{
				if (filter(node))
				{
					result.Add(node);
				}
			}
			return result;
		}
	}
}
