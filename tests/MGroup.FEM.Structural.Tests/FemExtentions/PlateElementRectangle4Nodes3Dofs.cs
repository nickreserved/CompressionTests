using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MGroup.Constitutive.Structural;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.DataStructures;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Entities;
using Microsoft.VisualStudio.TestPlatform.ObjectModel;
using TriangleNet.Topology;

namespace MGroup.FEM.Structural.Tests.Plates.FemExtentions
{
	/// <summary>
	/// Plate element with 4 nodes and 3 dofs per node: translation z, rotation around x, rotation around y. Follows Krichhoff-Love plate theory.
	/// </summary>
	public class PlateElementRectangle4Nodes3Dofs : IStructuralElementType
	{
		private static readonly IDofType[] nodalDOFTypes = new IDofType[3] 
		{ 
			StructuralDof.TranslationZ, StructuralDof.RotationX, StructuralDof.RotationY 
		};

		private static readonly IDofType[][] dofs = new IDofType[][] 
		{ 
			nodalDOFTypes, nodalDOFTypes, nodalDOFTypes, nodalDOFTypes 
		};

		private readonly double bendingStiffness;
		private readonly double poissonRatio;

		public PlateElementRectangle4Nodes3Dofs(
			IReadOnlyList<INode> nodes, double youngModulus, double poissonRatio, double thickness)
		{
			// Check that nodes create a rectangle
			if (nodes[0].Y != nodes[1].Y || nodes[2].Y != nodes[3].Y
				|| nodes[1].X != nodes[2].X || nodes[3].X != nodes[0].X
				|| nodes[1].X <= nodes[0].X || nodes[2].Y <= nodes[1].Y)
			{
				throw new ArgumentException("The nodes provided to this ractangular element do not form a rectangle.");
			}

			Nodes = nodes;
			this.poissonRatio = poissonRatio;
			bendingStiffness = youngModulus * Math.Pow(thickness, 3) / (12 * (1 - Math.Pow(poissonRatio, 2)));
		}

		public int ID { get; set; }

		public IReadOnlyList<INode> Nodes { get; }

		public int SubdomainID { get; set; } = 0;

		public CellType CellType => CellType.Quad4;

		public IElementDofEnumerator DofEnumerator { get; set; } = new GenericDofEnumerator();

		/// <summary>
		/// Calculates [Mx My Mxy] moments at a given point. Mx = bending moment around axis x, My = bending moment around axis y,
		/// Mxy = twisting moment.
		/// </summary>
		/// <param name="localCoords"></param>
		/// <param name="nodalDisplacements"></param>
		/// <returns></returns>
		public double[] CalcSectionMomentsAt(double[] localCoords, double[] nodalDisplacements)
		{
			Matrix invA = CalcInverseA();

			// Matrix vector 'β', with dimensions (3 x ndofs), needed to calculate curvatures
			double x = localCoords[0];
			double y = localCoords[1];
			var beta = Matrix.CreateFromArray(new double[,]
			{
				{ 0, 0, 0, 2, 0, 0, 6*x, 2*y, 0, 0, 6*x*y, 0 },
				{ 0, 0, 0, 0, 0, 2, 0, 0, 2 * x, 6 * y, 0, 6 * x * y },
				{ 0, 0, 0, 0, 2, 0, 0, 4 * x, 4 * y, 0, 6 * x * x, 6 * y * y },
			});

			// Curvatures
			var d = Vector.CreateFromArray(nodalDisplacements);
			Vector curvatures = beta * (invA * d);

			// Moments
			double Mx = bendingStiffness * (curvatures[0] + poissonRatio * curvatures[1]);
			double My = bendingStiffness * (curvatures[1] + poissonRatio * curvatures[0]);
			double Mxy = bendingStiffness * (1 - poissonRatio) / 2 * curvatures[2];

			return new double[] { Mx, My, Mxy };
		}

		public double CalcVerticalDisplacementAt(double[] localCoords, double[] nodalDisplacements)
		{
			Matrix invA = CalcInverseA();

			// Row vector 'x', with dimensions (1 x ndofs), needed to calculate the shape functions
			double x = localCoords[0];
			double y = localCoords[1];
			var vectorX = Vector.CreateFromArray(new double[] {
				1, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y,  y*y*y, x*x*x*y, x*y*y*y 
			});

			// Perform the interpolation
			var d = Vector.CreateFromArray(nodalDisplacements);
			double w = vectorX * (invA * d);
			return w;
		}

		public Tuple<double[], double[]> CalculateResponse(double[] localDisplacements) => throw new NotImplementedException();

		public double[] CalculateResponseIntegral() => throw new NotImplementedException();

		public double[] CalculateResponseIntegralForLogging(double[] localDisplacements) => throw new NotImplementedException();

		public Vector ConvertBodyForcesToNodal(double uniformLoadZ)
		{
			double a = 0.5 * (Nodes[1].X - Nodes[0].X);
			double b = 0.5 * (Nodes[2].Y - Nodes[1].Y);
			double q = uniformLoadZ;

			var result = Vector.CreateFromArray(new double[] {
				1, b / 3, a / 3, 1, b / 3, -a / 3, 1, -b / 3, -a / 3, 1, -b / 3, a / 3
			});
			result.ScaleIntoThis(q * a * b);

			return result;
		}

		public IMatrix DampingMatrix() => throw new NotImplementedException();

		public IReadOnlyList<IReadOnlyList<IDofType>> GetElementDofTypes() => dofs;

		public IEnumerable<double[]> IntegrateElementModelQuantities(IEnumerable<IElementModelQuantity<IStructuralDofType>> quantities) 
			=> throw new NotImplementedException();

		public IEnumerable<IEnumerable<double>> InterpolateElementModelQuantities(IEnumerable<IElementModelQuantity<IStructuralDofType>> quantities, IEnumerable<double[]> coordinates)
			=> throw new NotImplementedException();

		public IMatrix MassMatrix()
			=> throw new NotImplementedException();

		public IMatrix PhysicsMatrix() => StiffnessMatrix();

		public void SaveConstitutiveLawState(IHaveState externalState) => throw new NotImplementedException();

		public IMatrix StiffnessMatrix()
		{
			// Constants
			double a = 0.5 * (Nodes[1].X - Nodes[0].X);
			double b = 0.5 * (Nodes[2].Y - Nodes[1].Y);
			double p = Math.Pow(a / b, 2);
			double v = poissonRatio;

			// Coefficients of column 3
			double k33 = a * a * (80 / p + 16 - 16 * v);
			double k43 = a * (-60 / p - 6 + 6 * v);
			double k53 = 0;
			double k63 = a * a * (40 / p - 4 + 4 * v);
			double k73 = a * (-30 / p + 6 - 6 * v);
			double k83 = 0;
			double k93 = a * a * (20 / p + 4 - 4 * v);
			double k10_3 = a * (30 / p - 6 - 24 * v);
			double k11_3 = 0;
			double k12_3 = a * a * (40 / p - 16 + 16 * v);

			// Coefficients of column 2
			double k22 = b * b * (80 * p + 16 - 16 * v);
			double k32 = 60 * v * a * b;
			double k42 = b * (30 * p - 6 - 24 * v);
			double k52 = b * b * (40 * p - 16 + 16 * v);
			double k62 = -k53;
			double k72 = b * (-30 * p + 6 - 6 * v);
			double k82 = b * b * (20 * p + 4 - 4 * v);
			double k92 = k83;
			double k10_2 = b * (-60 * p - 6 + 6 * v);
			double k11_2 = b * b * (40 * p - 4 + 4 * v);
			double k12_2 = -k11_3;

			// Coefficients of column 1
			double k11 = 60 * p + 60 / p + 42 - 12 * v;
			double k21 = b * (60 * p + 6 + 24 * v);
			double k31 = a * (60 / p + 6 + 24 * v);
			double k41 = 30 * p - 60 / p - 42 + 12 * v;
			double k51 = k42;
			double k61 = -k43;
			double k71 = -30 * p - 30 / p + 42 - 12 * v;
			double k81 = -k72;
			double k91 = -k73;
			double k10_1 = -60 * p + 30 / p - 42 + 12 * v;
			double k11_1 = -k10_2;
			double k12_1 = k10_3;

			// Lower triangle of the stiffness matrix
			double[,] ke =
			{
				{ k11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ k21, k22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ k31, k32, k33, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ k41, k42, k43, k11, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ k51, k52, k53, k21, k22, 0, 0, 0, 0, 0, 0, 0 },
				{ k61, k62, k63, -k31, -k32, k33, 0, 0, 0, 0, 0, 0 },
				{ k71, k72, k73, k10_1, k10_2, -k10_3, k11, 0, 0, 0, 0, 0 },
				{ k81, k82, k83, k11_1, k11_2, -k11_3, -k21, k22, 0, 0, 0, 0 },
				{ k91, k92, k93, -k12_1, -k12_2, k12_3, -k31, k32, k33, 0, 0, 0 },
				{ k10_1, k10_2, k10_3, k71, k72, -k73, k41, -k42, -k43, k11, 0, 0 },
				{ k11_1, k11_2, k11_3, k81, k82, -k83, -k51, k52, k53, -k21, k22, 0 },
				{ k12_1, k12_2, k12_3, -k91, -k92, k93, -k61, k62, k63, k31, -k32, k33 }
			};

			//Transpose - copy from lower triangle to upper triangle
			for (int i = 0; i < 12; i++)
			{
				for (int j = i + 1; j < 12; j++)
				{
					ke[i, j] = ke[j, i];
				}
			}
			var result = Matrix.CreateFromArray(ke);
			//var result = TriangularLower.CreateFromArray(ke);

			// Finally multiply with D / (60 * a * b)
			result.ScaleIntoThis(bendingStiffness / (60 * a * b));
			return result;
		}

		/// <summary>
		/// Calculate the inverse of matrix 'A', needed for the shape functions
		/// </summary>
		private Matrix CalcInverseA()
		{
			double a = 0.5 * (Nodes[1].X - Nodes[0].X);
			double b = 0.5 * (Nodes[2].Y - Nodes[1].Y);
			double a2 = a * a;
			double a3 = a2 * a;
			double a4 = a3 * a;
			double b2 = b * b;
			double b3 = b2 * b;
			double b4 = b3 * b;

			var invA = Matrix.CreateFromArray(new double[,] {
				{  2*a3*b3, a3*b4, a4*b3, 2*a3*b3, a3*b4, -a4*b3, 2*a3*b3, -a3*b4, -a4*b3, 2*a3*b3, -a3*b4, a4*b3 },
				{ - 3 * a2*b3, -a2*b4, -a3*b3, 3 * a2*b3, a2*b4, -a3*b3, 3 * a2*b3, -a2*b4, -a3*b3, -3 * a2*b3, a2*b4, -a3*b3 },
				{ -3 * a3*b2, -a3*b3, -a4*b2, -3 * a3*b2, -a3*b3, a4*b2, 3 * a3*b2, -a3*b3, -a4*b2, 3 * a3*b2, -a3*b3, a4*b2 },
				{ 0, 0, -a2*b3, 0, 0, a2*b3, 0, 0, a2*b3, 0, 0, -a2*b3 },
				{ 4 * a2*b2, a2*b3, a3*b2, -4 * a2*b2, -a2*b3, a3*b2, 4 * a2*b2, -a2*b3, -a3*b2, -4 * a2*b2, a2*b3, -a3*b2 },
				{ 0, -a3*b2, 0, 0, -a3*b2, 0, 0, a3*b2, 0, 0, a3*b2, 0 },
				{ b3, 0, a*b3, -b3, 0, a*b3, -b3, 0, a*b3, b3, 0, a*b3 },
				{ 0, 0, a2*b2, 0, 0, -a2*b2, 0, 0, a2*b2, 0, 0, -a2*b2 },
				{ 0, a2*b2, 0, 0, -a2*b2, 0, 0, a2*b2, 0, 0, -a2*b2, 0 },
				{ a3, a3*b, 0, a3, a3*b, 0, -a3, a3*b, 0, -a3, a3*b, 0 },
				{ -b2, 0, -a*b2, b2, 0, -a*b2, -b2, 0, a*b2, b2, 0, a*b2 },
				{ -a2, -a2*b, 0, a2, a2*b, 0, -a2, a2*b, 0, a2, -a2*b, 0 },
			});
			invA.ScaleIntoThis(1 / (8 * a3 * b3));

			return invA;
		}
	}
}
