//
// ELL SWIG interface for module mfcc
//

%module(directors="1") mfcc
%feature("autodoc", "3");

%include "stdint.i"
%include "vector.i"

// Propagate python callback exceptions
%feature("director:except") {
    if ($error != NULL) {
        PyObject* ptype = nullptr;
        PyObject* pvalue = nullptr;
        PyObject* ptraceback = nullptr;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_Restore(ptype, pvalue, ptraceback);
        PyErr_Print();
        Py_Exit(1);
    }
}

%{
#include "featurizer_16k.i.h"
%}

%feature("director") MfccWrapper;


#if defined(SWIGPYTHON)
%pythoncode %{


_model_wrapper = None

def predict(inputData: 'numpy.ndarray') -> "numpy.ndarray":
    """Convenience function for calling the model directly without callbacks"""
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = MfccWrapper()

    if _model_wrapper.IsSteppable():
        raise Exception("You need to use the MfccWrapper directly because this model is steppable, which means the input is provided by a callback method")
        
    inputVector = FloatVector(inputData)
    output = _model_wrapper.Predict(inputVector)
    return np.array(output)

def reset():
    mfcc_Reset()


%}
#endif // defined(SWIGPYTHON)

%include "featurizer_16k.i.h"
%include "shape.i";
%inline %{
  ell::api::math::TensorShape get_default_input_shape() {
    TensorShape  s;
    mfcc_GetInputShape(0, &s);
    return ell::api::math::TensorShape{ s.rows, s.columns, s.channels };
  }
  ell::api::math::TensorShape get_default_output_shape() {
    TensorShape  s;
    mfcc_GetOutputShape(0, &s);
    return ell::api::math::TensorShape{ s.rows, s.columns, s.channels };
  }
%}

