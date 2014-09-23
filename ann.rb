
class InputNeuron

  attr_accessor :input
  attr_accessor :connected_output_neurons

  def initialize
    @input = nil
    @connected_output_neurons=[]
  end

  def output
    @input
  end
end

class Neuron
  attr_accessor :inputs, :weights, :error
  attr_accessor :connected_output_neurons

  def initialize(weights=[])
    @inputs = []
    @output = nil
    @connected_output_neurons=[]
    @weights = weights
    @error = nil
  end

  def output
    #if @output.nil?
      weighted_inputs = []
      @inputs.each_with_index {|n,i| weighted_inputs << n.output*@weights[i]}
      sum = weighted_inputs.inject(0, :+)
      @output = 1.0 / (1 + Math.exp(-sum))
    #end
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

  def weighted_error(input_neuron)
    index = @inputs.index(input_neuron)
    @error * weights[index]
  end

end

class Layer

  attr_accessor :name
  attr_reader :neurons

  def initialize(name, neuron_count, neuron_class)
    @neurons = []
    neuron_count.times do
      @neurons << neuron_class.new
    end
  end

  def neuron_count
    @neurons.count
  end
end

class NeuralNetwork

  attr_reader :hidden_layers
  attr_reader :output_layer

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

  def initialize(input_count=nil, hidden_neuron_count=nil, output_neuron_count=nil)
    @input_layer = nil
    @hidden_layers = []
    @output_layer = nil
    unless input_count.nil?
      build_nn(input_count, hidden_neuron_count, output_neuron_count)
    end
  end

  def create_input_layer(neuron_count)
    @input_layer = Layer.new('input', neuron_count, InputNeuron)
  end

  def create_hidden_layer(name, neuron_count)
    layer = Layer.new(name, neuron_count, Neuron)
    @hidden_layers << layer
    layer
  end

  def create_output_layer(neuron_count)
    @output_layer = Layer.new('output', neuron_count, Neuron)
  end

  def connect(from_layer, from_neuron_range, to_layer, to_neuron_range)
    neurons_to_connect = to_layer.neurons[to_neuron_range]
    from_layer.neurons[from_neuron_range].each do |neuron|
      neurons_to_connect.each do |neuron_to_connect|
        neuron_to_connect.inputs << neuron
        neuron.connected_output_neurons << neuron_to_connect
      end
    end
    neurons_to_connect.each do |neuron|
      neuron.weights = build_weights(neuron.inputs.count)
    end
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
    @input_layer.neurons.each_with_index do |n, i|
      n.input = inputs[i]
    end
    @output_layer.neurons.map {|n| n.output}
  end

  def calc_output_errors(expected_outputs)
    @output_layer.neurons.each_with_index do |neuron, i|
      neuron.error =
        (expected_outputs[i] - neuron.output) *  neuron.output * (1 - neuron.output)
    end
  end

  def calc_hidden_layer_errors()
    @hidden_layers.reverse.each do |layer|
      layer.neurons.each_with_index do |neuron, i|
        weighted_errors = neuron.connected_output_neurons.map {|on| on.weighted_error(neuron)}

        #weighted_errors = @output_neurons.map { |n| n.error*n.weights[i] }
        sum = weighted_errors.inject(0, :+)
        output = neuron.output
        neuron.error = output * (1.0 - output) * sum
      end
    end
  end

  def update_weights
    @hidden_layers.reverse.each do |layer|
      update_neurons_weights(layer.neurons)
    end
    update_neurons_weights(@output_layer.neurons)
  end

private

  def update_neurons_weights(neurons)
    neurons.each do |n|
      n.inputs.each_with_index do |input_neuron, i|
        n.weights[i] = n.weights[i] + 0.3 * n.error * input_neuron.output
      end
    end
  end


  def build_nn(input_count, hidden_neuron_count, output_neuron_count)
    l = self.create_input_layer(input_count)
    hl = self.create_hidden_layer('L1', hidden_neuron_count)
    ol = self.create_output_layer(output_neuron_count)
    self.connect(l, 0..-1, hl, 0..-1)
    self.connect(hl, 0..-1, ol, 0..-1)
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

550.times do
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
