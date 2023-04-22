classdef Gemm_To_GemmLayer1014 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
        fc1_action_bias
        fc1_action_weight
        fc1_state_bias
        fc1_state_weight
        fc2_bias
        fc2_weight
        fc3_bias
        fc3_weight
        fc4_bias
        fc4_weight
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = Gemm_To_GemmLayer1014(name, onnxParams)
            this.Name = name;
            this.NumInputs = 4;
            this.OutputNames = {'x21'};
            this.ONNXParams = onnxParams;
            this.fc1_action_bias = onnxParams.Learnables.fc1_action_bias;
            this.fc1_action_weight = onnxParams.Learnables.fc1_action_weight;
            this.fc1_state_bias = onnxParams.Learnables.fc1_state_bias;
            this.fc1_state_weight = onnxParams.Learnables.fc1_state_weight;
            this.fc2_bias = onnxParams.Learnables.fc2_bias;
            this.fc2_weight = onnxParams.Learnables.fc2_weight;
            this.fc3_bias = onnxParams.Learnables.fc3_bias;
            this.fc3_weight = onnxParams.Learnables.fc3_weight;
            this.fc4_bias = onnxParams.Learnables.fc4_bias;
            this.fc4_weight = onnxParams.Learnables.fc4_weight;
        end
        
        function [x21] = predict(this, onnx__Gemm_0, onnx__Gemm_1, onnx__Gemm_0NumDims, onnx__Gemm_1NumDims)
            if isdlarray(onnx__Gemm_0)
                onnx__Gemm_0 = stripdims(onnx__Gemm_0);
            end
            if isdlarray(onnx__Gemm_1)
                onnx__Gemm_1 = stripdims(onnx__Gemm_1);
            end
            onnx__Gemm_0NumDims = numel(onnx__Gemm_0NumDims);
            onnx__Gemm_1NumDims = numel(onnx__Gemm_1NumDims);
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.fc1_action_bias = this.fc1_action_bias;
            onnxParams.Learnables.fc1_action_weight = this.fc1_action_weight;
            onnxParams.Learnables.fc1_state_bias = this.fc1_state_bias;
            onnxParams.Learnables.fc1_state_weight = this.fc1_state_weight;
            onnxParams.Learnables.fc2_bias = this.fc2_bias;
            onnxParams.Learnables.fc2_weight = this.fc2_weight;
            onnxParams.Learnables.fc3_bias = this.fc3_bias;
            onnxParams.Learnables.fc3_weight = this.fc3_weight;
            onnxParams.Learnables.fc4_bias = this.fc4_bias;
            onnxParams.Learnables.fc4_weight = this.fc4_weight;
            [x21, x21NumDims] = Gemm_To_GemmFcn(onnx__Gemm_0, onnx__Gemm_1, onnx__Gemm_0NumDims, onnx__Gemm_1NumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {['as-is'], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {['as-is'], ['as-is']});
            if any(cellfun(@(A)isempty(A)||~isnumeric(A), {x21}))
                fprintf('Runtime error in network. The custom layer ''%s'' output an empty or non-numeric value.\n', 'Gemm_To_GemmLayer1014');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Gemm_To_GemmLayer1014'));
            end
            x21 = dlarray(single(x21), repmat('U', 1, max(2, x21NumDims)));
            if ~coder.target('MATLAB')
                x21 = extractdata(x21);
            end
        end
        
        function [x21] = forward(this, onnx__Gemm_0, onnx__Gemm_1, onnx__Gemm_0NumDims, onnx__Gemm_1NumDims)
            if isdlarray(onnx__Gemm_0)
                onnx__Gemm_0 = stripdims(onnx__Gemm_0);
            end
            if isdlarray(onnx__Gemm_1)
                onnx__Gemm_1 = stripdims(onnx__Gemm_1);
            end
            onnx__Gemm_0NumDims = numel(onnx__Gemm_0NumDims);
            onnx__Gemm_1NumDims = numel(onnx__Gemm_1NumDims);
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.fc1_action_bias = this.fc1_action_bias;
            onnxParams.Learnables.fc1_action_weight = this.fc1_action_weight;
            onnxParams.Learnables.fc1_state_bias = this.fc1_state_bias;
            onnxParams.Learnables.fc1_state_weight = this.fc1_state_weight;
            onnxParams.Learnables.fc2_bias = this.fc2_bias;
            onnxParams.Learnables.fc2_weight = this.fc2_weight;
            onnxParams.Learnables.fc3_bias = this.fc3_bias;
            onnxParams.Learnables.fc3_weight = this.fc3_weight;
            onnxParams.Learnables.fc4_bias = this.fc4_bias;
            onnxParams.Learnables.fc4_weight = this.fc4_weight;
            [x21, x21NumDims] = Gemm_To_GemmFcn(onnx__Gemm_0, onnx__Gemm_1, onnx__Gemm_0NumDims, onnx__Gemm_1NumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {['as-is'], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {['as-is'], ['as-is']});
            if any(cellfun(@(A)isempty(A)||~isnumeric(A), {x21}))
                fprintf('Runtime error in network. The custom layer ''%s'' output an empty or non-numeric value.\n', 'Gemm_To_GemmLayer1014');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Gemm_To_GemmLayer1014'));
            end
            x21 = dlarray(single(x21), repmat('U', 1, max(2, x21NumDims)));
            if ~coder.target('MATLAB')
                x21 = extractdata(x21);
            end
        end
    end
end

function [x21, x21NumDims, state] = Gemm_To_GemmFcn(onnx__Gemm_0, onnx__Gemm_1, onnx__Gemm_0NumDims, onnx__Gemm_1NumDims, params, varargin)
%GEMM_TO_GEMMFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 14
%
% Variable names in this function are taken from the original ONNX file.
%
% [X21] = Gemm_To_GemmFcn(ONNX__GEMM_0, ONNX__GEMM_1, PARAMS)
%			- Evaluates the imported ONNX network GEMM_TO_GEMMFCN with input(s)
%			ONNX__GEMM_0, ONNX__GEMM_1 and the imported network parameters in PARAMS. Returns
%			network output(s) in X21.
%
% [X21, STATE] = Gemm_To_GemmFcn(ONNX__GEMM_0, ONNX__GEMM_1, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Gemm_To_GemmFcn(ONNX__GEMM_0, ONNX__GEMM_1, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
%			- Specifies additional name-value pairs described below:
%
% 'Training'
% 			Boolean indicating whether the network is being evaluated for
%			prediction or training. If TRAINING is true, state variables
%			will be updated.
%
% 'InputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			 between the dimensions of the input data and the dimensions of
%			the ONNX model input. For example, the permutation from HWCN
%			(MATLAB standard) to NCHW (ONNX standard) uses the vector
%			[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for
%			more information about automatic permutation.
%
%			'none' - Input(s) are passed in the ONNX model format. See 'Inputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between input data dimensions and the expected
%			ONNX input dimensions.%
%			cell array - If the network has multiple inputs, each cell
%			contains 'auto', 'none', or a numeric vector.
%
% 'OutputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			between the dimensions of the output and a conventional MATLAB
%			dimension ordering. For example, the permutation from NC (ONNX
%			standard) to CN (MATLAB standard) uses the vector [2 1]. See
%			the documentation for IMPORTONNXFUNCTION for more information
%			about automatic permutation.
%
%			'none' - Return output(s) as given by the ONNX model. See 'Outputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between the ONNX output dimensions and the
%			desired output dimensions.%
%			cell array - If the network has multiple outputs, each cell
%			contains 'auto', 'none' or a numeric vector.
%
% Inputs:
% -------
% ONNX__GEMM_0, ONNX__GEMM_1
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  ONNX__GEMM_0:		[1, 3]				Type: FLOAT
%				  ONNX__GEMM_1:		[1, 1]				Type: FLOAT
%			  By default, the function will try to permute the input(s)
%			  into this dimension ordering. If the default is incorrect,
%			  use the 'InputDataPermutation' argument to control the
%			  permutation.
%
%
% PARAMS	- Network parameters returned by 'importONNXFunction'.
%
%
% Outputs:
% --------
% X21
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  X21:		[1, 1]				Type: FLOAT
%			  By default, the function will try to permute the output(s)
%			  from this dimension ordering into a conventional MATLAB
%			  ordering. If the default is incorrect, use the
%			  'OutputDataPermutation' argument to control the permutation.
%
% STATE		- (Optional) State variables. When TRAINING is true, these will
% 			  have been updated from the original values in PARAMS.State.
%
%
%  See also importONNXFunction

% Preprocess the input data and arguments:
[onnx__Gemm_0, onnx__Gemm_1, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(onnx__Gemm_0, onnx__Gemm_1, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'onnx__Gemm_0', 'onnx__Gemm_1'}, {onnx__Gemm_0, onnx__Gemm_1}, [onnx__Gemm_0NumDims onnx__Gemm_1NumDims]);
% Call the top-level graph function:
[x21, x21NumDims, state] = Gemm_To_GemmGraph1000(onnx__Gemm_0, onnx__Gemm_1, NumDims.onnx__Gemm_0, NumDims.onnx__Gemm_1, Vars, NumDims, Training, params.State);
% Postprocess the output data
[x21] = postprocessOutput(x21, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [x21, x21NumDims1013, state] = Gemm_To_GemmGraph1000(onnx__Gemm_0, onnx__Gemm_1, onnx__Gemm_0NumDims1011, onnx__Gemm_1NumDims1012, Vars, NumDims, Training, state)
% Function implementing the graph 'Gemm_To_GemmGraph1000'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.onnx__Gemm_0 = onnx__Gemm_0;
NumDims.onnx__Gemm_0 = onnx__Gemm_0NumDims1011;
Vars.onnx__Gemm_1 = onnx__Gemm_1;
NumDims.onnx__Gemm_1 = onnx__Gemm_1NumDims1012;

% Execute the operators:
% Gemm:
[A, B, C, alpha, beta, NumDims.x_fc1_state_Gemm_out] = prepareGemmArgs(Vars.onnx__Gemm_0, Vars.fc1_state_weight, Vars.fc1_state_bias, Vars.Gemmalpha1001, Vars.Gemmbeta1002, 0, 1, NumDims.fc1_state_bias);
Vars.x_fc1_state_Gemm_out = alpha*B*A + beta*C;

% Relu:
Vars.x_relu1_state_Relu_o = relu(Vars.x_fc1_state_Gemm_out);
NumDims.x_relu1_state_Relu_o = NumDims.x_fc1_state_Gemm_out;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_fc1_action_Gemm_ou] = prepareGemmArgs(Vars.onnx__Gemm_1, Vars.fc1_action_weight, Vars.fc1_action_bias, Vars.Gemmalpha1003, Vars.Gemmbeta1004, 0, 1, NumDims.fc1_action_bias);
Vars.x_fc1_action_Gemm_ou = alpha*B*A + beta*C;

% Relu:
Vars.x_relu1_action_Relu_ = relu(Vars.x_fc1_action_Gemm_ou);
NumDims.x_relu1_action_Relu_ = NumDims.x_fc1_action_Gemm_ou;

% Concat:
[Vars.x_Concat_output_0, NumDims.x_Concat_output_0] = onnxConcat(1, {Vars.x_relu1_state_Relu_o, Vars.x_relu1_action_Relu_}, [NumDims.x_relu1_state_Relu_o, NumDims.x_relu1_action_Relu_]);

% Gemm:
[A, B, C, alpha, beta, NumDims.x_fc2_Gemm_output_0] = prepareGemmArgs(Vars.x_Concat_output_0, Vars.fc2_weight, Vars.fc2_bias, Vars.Gemmalpha1005, Vars.Gemmbeta1006, 0, 1, NumDims.fc2_bias);
Vars.x_fc2_Gemm_output_0 = alpha*B*A + beta*C;

% Relu:
Vars.x_relu2_Relu_output_ = relu(Vars.x_fc2_Gemm_output_0);
NumDims.x_relu2_Relu_output_ = NumDims.x_fc2_Gemm_output_0;

% Gemm:
[A, B, C, alpha, beta, NumDims.x_fc3_Gemm_output_0] = prepareGemmArgs(Vars.x_relu2_Relu_output_, Vars.fc3_weight, Vars.fc3_bias, Vars.Gemmalpha1007, Vars.Gemmbeta1008, 0, 1, NumDims.fc3_bias);
Vars.x_fc3_Gemm_output_0 = alpha*B*A + beta*C;

% Relu:
Vars.x_relu3_Relu_output_ = relu(Vars.x_fc3_Gemm_output_0);
NumDims.x_relu3_Relu_output_ = NumDims.x_fc3_Gemm_output_0;

% Gemm:
[A, B, C, alpha, beta, NumDims.x21] = prepareGemmArgs(Vars.x_relu3_Relu_output_, Vars.fc4_weight, Vars.fc4_bias, Vars.Gemmalpha1009, Vars.Gemmbeta1010, 0, 1, NumDims.fc4_bias);
Vars.x21 = alpha*B*A + beta*C;

% Set graph output arguments from Vars and NumDims:
x21 = Vars.x21;
x21NumDims1013 = NumDims.x21;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(onnx__Gemm_0, onnx__Gemm_1, numDataOutputs, params, varargin)
% Function to validate inputs to Gemm_To_GemmFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'onnx__Gemm_0', isValidArrayInput);
addRequired(p, 'onnx__Gemm_1', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, onnx__Gemm_0, onnx__Gemm_1, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,2);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [onnx__Gemm_0, onnx__Gemm_1, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(onnx__Gemm_0, onnx__Gemm_1, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(onnx__Gemm_0, onnx__Gemm_1, 1, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {onnx__Gemm_0, onnx__Gemm_1}));
% Make the input variables into unlabelled dlarrays:
onnx__Gemm_0 = makeUnlabeledDlarray(onnx__Gemm_0);
onnx__Gemm_1 = makeUnlabeledDlarray(onnx__Gemm_1);
% Permute inputs if requested:
onnx__Gemm_0 = permuteInputVar(onnx__Gemm_0, inputDataPerms{1}, 2);
onnx__Gemm_1 = permuteInputVar(onnx__Gemm_1, inputDataPerms{2}, 2);
end

function [x21] = postprocessOutput(x21, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(x21)
        x21 = extractdata(x21);
    end
end
% Permute outputs if requested:
x21 = permuteOutputVar(x21, outputDataPerms{1}, 2);
end


%% dlarray functions implementing ONNX operators:

function [Y, numDimsY] = onnxConcat(ONNXAxis, XCell, numDimsXArray)
% Concatentation that treats all empties the same. Necessary because
% dlarray.cat does not allow, for example, cat(1, 1x1, 1x0) because the
% second dimension sizes do not match.
numDimsY = numDimsXArray(1);
XCell(cellfun(@isempty, XCell)) = [];
if isempty(XCell)
    Y = dlarray([]);
else
    if ONNXAxis<0
        ONNXAxis = ONNXAxis + numDimsY;
    end
    DLTAxis = numDimsY - ONNXAxis;
    Y = cat(DLTAxis, XCell{:});
end
end

function [A, B, C, alpha, beta, numDimsY] = prepareGemmArgs(A, B, C, alpha, beta, transA, transB, numDimsC)
% Prepares arguments for implementing the ONNX Gemm operator
if transA
    A = A';
end
if transB
    B = B';
end
if numDimsC < 2
    C = C(:);   % C can be broadcast to [N M]. Make C a col vector ([N 1])
end
numDimsY = 2;
% Y=B*A because we want (AB)'=B'A', and B and A are already transposed.
end

%% Utility functions:

function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.
if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end

function checkInputSize(inputShape, expectedShape, inputName)

if numel(expectedShape)==0
    % The input is a scalar
    if ~isequal(inputShape, [1 1])
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, "[1,1]", inputSizeStr));
    end
elseif numel(expectedShape)==1
    % The input is a vector
    if ~shapeIsColumnVector(inputShape) || ~iSizesMatch({inputShape(1)}, expectedShape)
        expectedShape{2} = 1;
        expectedSizeStr = makeSizeString(expectedShape);
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
else
    % The input has 2 dimensions or more
    
    % The input dimensions have been reversed; flip them back to compare to the
    % expected ONNX shape.
    inputShape = fliplr(inputShape);
    
    % If the expected shape has fewer dims than the input shape, error.
    if numel(expectedShape) < numel(inputShape)
        expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
        error(message('nnet_cnn_onnx:onnx:InputHasGreaterNDims', inputName, expectedSizeStr));
    end
    
    % Prepad the input shape with trailing ones up to the number of elements in
    % expectedShape
    inputShape = num2cell([ones(1, numel(expectedShape) - length(inputShape)) inputShape]);
    
    % Find the number of variable size dimensions in the expected shape
    numVariableInputs = sum(cellfun(@(x) isa(x, 'char') || isa(x, 'string'), expectedShape));
    
    % Find the number of input dimensions that are not in the expected shape
    % and cannot be represented by a variable dimension
    nonMatchingInputDims = setdiff(string(inputShape), string(expectedShape));
    numNonMatchingInputDims  = numel(nonMatchingInputDims) - numVariableInputs;
    
    expectedSizeStr = makeSizeString(expectedShape);
    inputSizeStr = makeSizeString(inputShape);
    if numNonMatchingInputDims == 0 && ~iSizesMatch(inputShape, expectedShape)
        % The actual and expected input dimensions match, but in
        % a different order. The input needs to be permuted.
        error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));
    elseif numNonMatchingInputDims > 0
        % The actual and expected input sizes do not match.
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
end
end

function doesMatch = iSizesMatch(inputShape, expectedShape)
% Check whether the input and expected shapes match, in order.
% Size elements match if (1) the elements are equal, or (2) the expected
% size element is a variable (represented by a character vector or string)
doesMatch = true;
for i=1:numel(inputShape)
    if ~(isequal(inputShape{i},expectedShape{i}) || ischar(expectedShape{i}) || isstring(expectedShape{i}))
        doesMatch = false;
        return
    end
end
end

function sizeStr = makeSizeString(shape)
sizeStr = strjoin(["[", strjoin(string(shape), ","), "]"], "");
end

function isVec = shapeIsColumnVector(shape)
if numel(shape) == 2 && shape(2) == 1
    isVec = true;
else
    isVec = false;
end
end
function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray
if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if isinteger(X)
        % Make ints double so they can combine with anything without
        % reducing precision
        X = double(X);
    end
    X = dlarray(X);
end
end

function [Vars, NumDims] = packageVariables(params, inputNames, inputValues, inputNumDims)
% inputNames, inputValues are cell arrays. inputRanks is a numeric vector.
Vars = appendStructs(params.Learnables, params.Nonlearnables, params.State);
NumDims = params.NumDimensions;
% Add graph inputs
for i = 1:numel(inputNames)
    Vars.(inputNames{i}) = inputValues{i};
    NumDims.(inputNames{i}) = inputNumDims(i);
end
end

function X = permuteInputVar(X, userDataPerm, onnxNDims)
% Returns reverse-ONNX ordering
if onnxNDims == 0
    return;
elseif onnxNDims == 1 && isvector(X)
    X = X(:);
    return;
elseif isnumeric(userDataPerm)
    % Permute into reverse ONNX ordering
    if numel(userDataPerm) ~= onnxNDims
        error(message('nnet_cnn_onnx:onnx:InputPermutationSize', numel(userDataPerm), onnxNDims));
    end
    perm = fliplr(userDataPerm);
elseif isequal(userDataPerm, 'auto') && onnxNDims == 4
    % Permute MATLAB HWCN to reverse onnx (WHCN)
    perm = [2 1 3 4];
elseif isequal(userDataPerm, 'as-is')
    % Do not permute the input
    perm = 1:ndims(X);
else
    % userDataPerm is either 'none' or 'auto' with no default, which means
    % it's already in onnx ordering, so just make it reverse onnx
    perm = max(2,onnxNDims):-1:1;
end
X = permute(X, perm);
end

function Y = permuteOutputVar(Y, userDataPerm, onnxNDims)
switch onnxNDims
    case 0
        perm = [];
    case 1
        if isnumeric(userDataPerm)
            % Use the user's permutation because Y is a column vector which
            % already matches ONNX.
            perm = userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            % Treat the 1D onnx vector as a 2D column and transpose it
            perm = [2 1];
        else
            % userDataPerm is 'none'. Leave Y alone because it already
            % matches onnx.
            perm = [];
        end
    otherwise
        % ndims >= 2
        if isnumeric(userDataPerm)
            % Use the inverse of the user's permutation. This is not just the
            % flip of the permutation vector.
            perm = onnxNDims + 1 - userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            if onnxNDims == 2
                % Permute reverse ONNX CN to DLT CN (do nothing)
                perm = [];
            elseif onnxNDims == 4
                % Permute reverse onnx (WHCN) to MATLAB HWCN
                perm = [2 1 3 4];
            else
                % User wants the output in ONNX ordering, so just reverse it from
                % reverse onnx
                perm = onnxNDims:-1:1;
            end
        elseif isequal(userDataPerm, 'as-is')
            % Do not permute the input
            perm = 1:ndims(Y);
        else
            % userDataPerm is 'none', so just make it reverse onnx
            perm = onnxNDims:-1:1;
        end
end
if ~isempty(perm)
    Y = permute(Y, perm);
end
end

function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in t.
for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end
