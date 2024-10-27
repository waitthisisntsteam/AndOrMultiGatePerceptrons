using System.Transactions;

namespace AndOrMultiGatePerceptrons
{
    internal class Program
    {
        // Activation Functions
        static double Sigmoid(double input) => 1 / (1 + Math.Pow(double.E, -input));
        static double SigmoidD(double input) => Sigmoid(input) * (1 - Sigmoid(input));

        static double TanH(double input) => (Math.Pow(double.E, input) - Math.Pow(double.E, -input)) / (Math.Pow(double.E, input) + Math.Pow(double.E, -input));
        static double TanHD(double input) => 1 - Math.Pow(TanH(input), 2);

        // Error Functions
        static double MeanSquared(double input, double desiredOutput) => Math.Pow(desiredOutput - input, 2);
        static double MeanSquaredD(double input, double desiredOutput) => 2 * (desiredOutput - input);

        static void Main(string[] args)
        {
            Perceptron gate = new Perceptron(3, 0.01, new ErrorFunction(MeanSquared, MeanSquaredD), new ActivationFunction(TanH, TanHD));
            gate.Randomize(new Random(), 0, 2);
            double[][] input = [ [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
            double[] desiredOutputs = [0, 1, 1, 1, 0, 0, 0, 1];

            double currentError = gate.GetError(input, desiredOutputs);
            while (true)
            {
                currentError = gate.TrainORGate(input, desiredOutputs, currentError);
                Console.WriteLine(currentError);

                var d = gate.UserComputeWithActivation(input);
                ;
            }
        }
    }
}
