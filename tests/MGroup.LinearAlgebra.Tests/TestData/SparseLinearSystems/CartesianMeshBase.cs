using Compression.tests.MGroup.LinearAlgebra.Tests.Utilities;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Meshes.Structured;
using System.Text;

namespace Compression.tests.MGroup.LinearAlgebra.Tests.TestData.SparseLinearSystems
{
    abstract public class CartesianMeshBase : IStructuredMesh
    {
        protected readonly int[] numElementsPerAxis;
        protected readonly int[] numNodesPerAxis;
        protected readonly double[] lengthPerAxis;
        protected readonly double[] nodeDistancePerAxis;

        public CartesianMeshBase(int[] numElementsPerAxis, double[] lengthPerAxis)
        {
            this.numElementsPerAxis = numElementsPerAxis;
            this.lengthPerAxis = lengthPerAxis;

            numNodesPerAxis = new int[Dimension];
            nodeDistancePerAxis = new double[Dimension];
            NumNodesTotal = NumElementsTotal = 1;
            for (int i = 0; i < Dimension; ++i)
            {
                NumElementsTotal *= numElementsPerAxis[i];
                numNodesPerAxis[i] = numElementsPerAxis[i] + 1;
                nodeDistancePerAxis[i] = lengthPerAxis[i] / numElementsPerAxis[i];
                NumNodesTotal *= numNodesPerAxis[i];
            }
        }

        public int Dimension { get => numElementsPerAxis.Length; }
        public int NumElementsTotal { get; protected set; }
        public int[] NumElementsPerAxis => ArrayUtilities.Copy(numElementsPerAxis);
        public int[] NumNodesPerAxis => ArrayUtilities.Copy(numNodesPerAxis);
        public double[] LengthPerAxis => ArrayUtilities.Copy(lengthPerAxis);
        public double[] NodeDistancePerAxis => ArrayUtilities.Copy(nodeDistancePerAxis);

        public int NumElementsOnAxis(int axis) => numElementsPerAxis[axis];
        public int NumNodesOnAxis(int axis) => numNodesPerAxis[axis];
        public double LengthOnAxis(int axis) => lengthPerAxis[axis];
        public double NodeDistanceOnAxis(int axis) => nodeDistancePerAxis[axis];

        abstract public int[] GetNodeIdsOfElement(int[] elementIdx);

        public double[] GetNodeCoordinates(int[] nodeIdx)
        {
            double[] coords = new double[Dimension];
            for (int i = 0; i < Dimension; ++i)
                coords[i] = nodeIdx[i] * nodeDistancePerAxis[i];
            return coords;
        }


        private void CheckElementId(int elementId)
        {
            if (elementId < 0 || elementId >= NumElementsTotal)
            {
                throw new ArgumentException(
                    $"Invalid element id={elementId}. It must belong to the interval [0, {NumElementsTotal})");
            }
        }

        private void CheckElementIndex(int[] elementIdx)
        {
            for (int i = 0; i < Dimension; ++i)
                if (elementIdx[i] < 0 || elementIdx[i] >= numElementsPerAxis[i])
                {
                    var msg = new StringBuilder("There is no element with index: (");
                    msg.Append(elementIdx[0]);
                    for (int j = 1; j < elementIdx.Length; ++j)
                    {
                        msg.Append(", ");
                        msg.Append(elementIdx[j]);
                    }
                    msg.Append(")");
                    throw new ArgumentException(msg.ToString());
                }
        }

        private void CheckNodeId(int nodeId)
        {
            if (nodeId < 0 || nodeId >= NumNodesTotal)
            {
                throw new ArgumentException(
                    $"Invalid node id={nodeId}. It must belong to the interval [0, {NumNodesTotal})");
            }
        }

        private void CheckNodeIndex(int[] nodeIdx)
        {
            for (int i = 0; i < Dimension; ++i)
                if (nodeIdx[i] < 0 || nodeIdx[i] >= numNodesPerAxis[i])
                {
                    var msg = new StringBuilder("There is no node with index: (");
                    msg.Append(nodeIdx[0]);
                    for (int j = 1; j < nodeIdx.Length; ++j)
                    {
                        msg.Append(", ");
                        msg.Append(nodeIdx[j]);
                    }
                    msg.Append(")");
                    throw new ArgumentException(msg.ToString());
                }
        }


        public abstract CellType CellType { get; }
        public abstract double[] MinCoordinates { get; }
        public abstract double[] MaxCoordinates { get; }
        public abstract int[] NumNodes { get; }
        public abstract int NumNodesPerElement { get; }
        public int NumNodesTotal { get; private set; }

        public IEnumerable<(int elementID, int[] nodeIDs)> EnumerateElements()
        {
            for (int i = 0; i < NumElementsTotal; ++i)
                yield return (i, GetNodeIdsOfElement(GetElementIdx(i)));
        }
        public IEnumerable<(int nodeID, double[] coordinates)> EnumerateNodes()
        {
            for (int i = 0; i < NumNodesTotal; ++i)
                yield return (i, GetNodeCoordinates(GetNodeIdx(i)));
        }

        public abstract int[] GetElementConnectivity(int[] elementIdx);
        public abstract int[] GetElementConnectivity(int elementID);

        public abstract int GetElementID(int[] elementIdx);
        public abstract int[] GetElementIdx(int elementID);

        public abstract int GetNodeID(int[] nodeIdx);
        public abstract int[] GetNodeIdx(int nodeID);
    }
}
