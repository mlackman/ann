
class Neuron
  attr_accessor :inputs, :weights, :error

  def initialize(weights=[])
    @inputs = []
    @output = nil
    @weights = weights
    @error = nil
  end

  def output
    if @output.nil?
      weighted_inputs = []
      @inputs.each_with_index {|v,i| weighted_inputs << v*@weights[i]}
      sum = weighted_inputs.inject(0, :+)
      @output = 1.0 / (1 + Math.exp(-sum))
    end
    @output
  end

  def inputs=(inputs)
    @output=nil
    @inputs = inputs
  end

  def weights=(weights)
    @output = nil
    @weights = weights
  end

end

class NeuralNetwork

  attr_reader :hidden_neurons, :output_neurons

  def self.build(hidden_neurons_weights, output_neuron_weights)
    nn = self.new(hidden_neurons_weights[0].count, hidden_neurons_weights.count, output_neuron_weights.count)
    nn.hidden_neurons.each_with_index do |n, i|
      n.weights = hidden_neurons_weights[i]
    end
    nn.output_neurons.each_with_index do |n, i|
      n.weights = output_neuron_weights[i]
    end
    nn
  end

  def initialize(input_count, hidden_neuron_count, output_neuron_count)
    build_nn(input_count, hidden_neuron_count, output_neuron_count)
  end

  def print_hidden_neurons
    print_neurons(@hidden_neurons)
  end

  def print_neurons(neurons)
    neurons.each {|n| p n}
  end

  def print_hidden_neurons_weights
    print_neurons_weights(@hidden_neurons)
  end

  def print_output_neurons_weights
    print_neurons_weights(@output_neurons)
  end

  def print_neurons_weights(neurons)
    neurons.each do |n|
      p n.weights
    end
  end


  def evaluate(inputs)
    hidden_neurons_outputs = []
    @hidden_neurons.each do |hidden_neuron|
      hidden_neuron.inputs = inputs
      hidden_neurons_outputs << hidden_neuron.output
    end

    output_neurons_outputs = []
    @output_neurons.each do |output_neuron|
      output_neuron.inputs = hidden_neurons_outputs
      output_neurons_outputs << output_neuron.output
    end
    output_neurons_outputs
  end

  def calc_output_errors(expected_outputs)
    @output_neurons.each_with_index do |neuron, i|
      neuron.error =
        (expected_outputs[i] - neuron.output) *  neuron.output * (1 - neuron.output)
    end
  end

  def calc_hidden_layer_errors()
    @hidden_neurons.each_with_index do |hidden_neuron, i|
      weighted_errors = @output_neurons.map { |n| n.error*n.weights[i] }
      sum = weighted_errors.inject(0, :+)
      hout = hidden_neuron.output
      hidden_neuron.error = hout * (1.0 - hout) * sum
    end
  end

  def update_weights
    update_neurons_weights(@hidden_neurons)
    update_neurons_weights(@output_neurons)
  end

private

  def update_neurons_weights(neurons)
    neurons.each do |n|
      n.inputs.each_with_index do |input, i|
        n.weights[i] = n.weights[i] + 0.3 * n.error * input
      end
    end
  end


  def build_nn(input_count, hidden_neuron_count, output_neuron_count)
    hidden_neurons = []
    output_neurons = []
    hidden_neuron_count.times do
      weights = build_weights(input_count)
      hidden_neurons << Neuron.new(weights)
    end
    output_neuron_count.times do
      output_neurons << Neuron.new(build_weights(hidden_neuron_count))
    end
    @hidden_neurons = hidden_neurons
    @output_neurons = output_neurons
  end

  def build_weights(count)
    weights = []
    count.times { weights << (rand * 2 - 1)/5.0 }
    weights
  end
end

def to_inputs(s, count)
  s = "" if s.nil?
  s.upcase!
  s = s.split.join(" ").split("")
  s = s[0..count-1]
  s = s.map {|ch| ch.ord/255.0 }
  s = [] if s == ""
  (count - s.length).times do
    s << 0
  end
  s
end

nn = NeuralNetwork.new(1, 4, 1)


training = []
20.times do
  v = rand
  e = 0.0
  if v > 0.5
    e = 1.0
  end
  training << [v, e]
end

150.times do
  success = 0
  training.each do |v,e|
    output = nn.evaluate([v])[0]
    if e == 1.0
      success += 1 if output > 0.5
    else
      success += 1 if output < 0.5
    end
    #p "training: #{v}, expecting #{e}, got #{nn.evaluate([v])}"

    nn.calc_output_errors([e])
    nn.calc_hidden_layer_errors
    nn.update_weights

  end
  p "#{success.to_f/training.count.to_f*100}"
end


nn.print_hidden_neurons_weights
p "Output neurons"
nn.print_output_neurons_weights
