using System.Transactions;

namespace AndOrMultiGatePerceptrons
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ActivationErorrFormulas activationErorrFormulas = new ActivationErorrFormulas();
            Perceptron gate = new Perceptron(3, 0.01, new ErrorFunction(activationErorrFormulas.MeanSquared, activationErorrFormulas.MeanSquaredD), new ActivationFunction(activationErorrFormulas.TanH, activationErorrFormulas.TanHD));
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
