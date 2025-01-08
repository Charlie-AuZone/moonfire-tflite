extern crate num;
#[macro_use]
extern crate num_derive;

// Copyright (C) 2020 Scott Lamb <slamb@slamb.org>
// SPDX-License-Identifier: Apache-2.0

use std::convert::TryFrom;
use std::ffi::{c_void, CStr};
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::ptr;

use delegate::Delegate;

#[cfg(feature = "edgetpu")]
pub mod edgetpu;

pub mod delegate;

// Opaque types from the C interface.
// https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
#[repr(C)]
pub struct TfLiteDelegate {
    _private: [u8; 0],
}
#[repr(C)]
struct TfLiteInterpreter {
    _private: [u8; 0],
}
#[repr(C)]
struct TfLiteInterpreterOptions {
    _private: [u8; 0],
}
#[repr(C)]
struct TfLiteModel {
    _private: [u8; 0],
}
#[repr(C)]
pub struct Tensor {
    _private: [u8; 0],
} // aka TfLiteTensor

// Type, aka TfLiteType
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub enum TensorType {
    NoType = 0,
    Float32 = 1,
    Int32 = 2,
    UInt8 = 3,
    Int64 = 4,
    String = 5,
    Bool = 6,
    Int16 = 7,
    Complex64 = 8,
    Int8 = 9,
    Float16 = 10,
}

#[derive(FromPrimitive, Debug)]
pub enum TfLiteStatusEnum {
    /// Success
    TfLiteOk = 0,

    /// Generally referring to an error in the runtime (i.e. interpreter)
    TfLiteError = 1,

    /// Generally referring to an error from a TfLiteDelegate itself.
    TfLiteDelegateError = 2,

    /// Generally referring to an error in applying a delegate due to
    /// incompatibility between runtime and delegate, e.g., this error is returned
    /// when trying to apply a TF Lite delegate onto a model graph that's already
    /// immutable.
    TfLiteApplicationError = 3,

    /// Generally referring to serialized delegate data not being found.
    /// See tflite::delegates::Serialization.
    TfLiteDelegateDataNotFound = 4,

    /// Generally referring to data-writing issues in delegate serialization.
    /// See tflite::delegates::Serialization.
    TfLiteDelegateDataWriteError = 5,

    /// Generally referring to data-reading issues in delegate serialization.
    /// See tflite::delegates::Serialization.
    TfLiteDelegateDataReadError = 6,

    /// Generally referring to issues when the TF Lite model has ops that cannot
    /// be resolved at runtime. This could happen when the specific op is not
    /// registered or built with the TF Lite framework.
    TfLiteUnresolvedOps = 7,

    /// Generally referring to invocation cancelled by the user.
    /// See `interpreter::Cancel`.
    // TODO(b/194915839): Implement `interpreter::Cancel`.
    // TODO(b/250636993): Cancellation triggered by `SetCancellationFunction`
    // should also return this status code.
    TfLiteCancelled = 8,

    // This status is returned by Prepare when the output shape cannot be
    // determined but the size of the output tensor is known. For example, the
    // output of reshape is always the same size as the input. This means that
    // such ops may be
    // done in place.
    TfLiteOutputShapeNotKnown = 9,
}

#[derive(Copy, Clone)]
#[repr(transparent)]
struct TfLiteStatus(libc::c_int);

// #[link(name = "tensorflowlite_c")
extern "C" {
    fn TfLiteModelCreate(model_data: *const u8, model_size: usize) -> *mut TfLiteModel;
    fn TfLiteModelDelete(model: *mut TfLiteModel);

    fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    fn TfLiteInterpreterOptionsDelete(interpreter: *mut TfLiteInterpreterOptions);
    fn TfLiteInterpreterOptionsAddDelegate(
        options: *mut TfLiteInterpreterOptions,
        delegate: *mut TfLiteDelegate,
    );

    fn TfLiteInterpreterCreate(
        model: *const TfLiteModel,
        options: *const TfLiteInterpreterOptions,
    ) -> *mut TfLiteInterpreter;
    fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;
    fn TfLiteInterpreterGetInputTensorCount(interpreter: *const TfLiteInterpreter) -> libc::c_int;
    fn TfLiteInterpreterGetInputTensor(
        interpreter: *const TfLiteInterpreter,
        input_index: i32,
    ) -> *mut Tensor;
    fn TfLiteInterpreterInvoke(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;
    fn TfLiteInterpreterGetOutputTensorCount(interpreter: *const TfLiteInterpreter) -> libc::c_int;
    fn TfLiteInterpreterGetOutputTensor(
        interpreter: *const TfLiteInterpreter,
        output_index: i32,
    ) -> *const Tensor;

    fn TfLiteTensorType(tensor: *const Tensor) -> TensorType;
    fn TfLiteTensorNumDims(tensor: *const Tensor) -> i32;
    fn TfLiteTensorDim(tensor: *const Tensor, dim_index: i32) -> i32;
    fn TfLiteTensorByteSize(tensor: *const Tensor) -> usize;
    fn TfLiteTensorData(tensor: *const Tensor) -> *mut c_void;
    fn TfLiteTensorName(tensor: *const Tensor) -> *const c_char;

    fn TfLiteTypeGetName(type_: TensorType) -> *const c_char;
}

impl TfLiteStatus {
    fn to_result(self) -> Result<(), String> {
        let status = match num::FromPrimitive::from_i32(self.0) {
            Some(v) => v,
            None => return Err(format!("Unknown TfLiteStatus: {}", self.0)),
        };
        match status {
            TfLiteStatusEnum::TfLiteOk => Ok(()),
            _ => Err(format!("{:?}", status)),
        }
    }
}

pub struct InterpreterBuilder<'a> {
    options: ptr::NonNull<TfLiteInterpreterOptions>,
    owned_delegates: Vec<Delegate>,
    _delegate_refs: PhantomData<&'a ()>,
}

impl<'a> InterpreterBuilder<'a> {
    pub fn new() -> Self {
        Self {
            options: ptr::NonNull::new(unsafe { TfLiteInterpreterOptionsCreate() }).unwrap(),
            owned_delegates: Vec::new(),
            _delegate_refs: PhantomData,
        }
    }

    pub fn add_borrowed_delegate(&mut self, d: &'a Delegate) {
        unsafe { TfLiteInterpreterOptionsAddDelegate(self.options.as_ptr(), d.delegate.as_ptr()) }
    }

    pub fn add_owned_delegate(&mut self, d: Delegate) {
        unsafe { TfLiteInterpreterOptionsAddDelegate(self.options.as_ptr(), d.delegate.as_ptr()) }
        self.owned_delegates.push(d);
    }

    pub fn build(mut self, model: &Model) -> Result<Interpreter<'a>, String> {
        let interpreter =
            unsafe { TfLiteInterpreterCreate(model.ptr.as_ptr(), self.options.as_ptr()) };
        let interpreter = Interpreter {
            interpreter: ptr::NonNull::new(interpreter)
                .ok_or("TfLiteInterpreterCreate returned NULL")?,
            _owned_delegates: std::mem::replace(&mut self.owned_delegates, Vec::new()),
            _delegate_refs: PhantomData,
        };
        unsafe { TfLiteInterpreterAllocateTensors(interpreter.interpreter.as_ptr()) }
            .to_result()?;
        Ok(interpreter)
    }
}

impl<'a> Drop for InterpreterBuilder<'a> {
    fn drop(&mut self) {
        unsafe { TfLiteInterpreterOptionsDelete(self.options.as_ptr()) };
    }
}

pub struct Interpreter<'a> {
    interpreter: ptr::NonNull<TfLiteInterpreter>,
    _owned_delegates: Vec<Delegate>,
    _delegate_refs: PhantomData<&'a ()>,
}

impl<'a> Interpreter<'a> {
    pub fn builder() -> InterpreterBuilder<'a> {
        InterpreterBuilder::new()
    }

    pub fn invoke(&mut self) -> Result<(), String> {
        unsafe { TfLiteInterpreterInvoke(self.interpreter.as_ptr()) }.to_result()
    }

    pub fn inputs(&self) -> InputTensors {
        let len = usize::try_from(unsafe {
            TfLiteInterpreterGetInputTensorCount(self.interpreter.as_ptr())
        })
        .unwrap();
        InputTensors {
            interpreter: self,
            len,
        }
    }

    pub fn inputs_mut(&mut self) -> InputTensorsMut {
        let len = usize::try_from(unsafe {
            TfLiteInterpreterGetInputTensorCount(self.interpreter.as_ptr())
        })
        .unwrap();
        InputTensorsMut {
            interpreter: self,
            len,
        }
    }

    pub fn outputs(&self) -> OutputTensors {
        let len = usize::try_from(unsafe {
            TfLiteInterpreterGetOutputTensorCount(self.interpreter.as_ptr())
        })
        .unwrap();
        OutputTensors {
            interpreter: self,
            len,
        }
    }
}

impl<'a> Drop for Interpreter<'a> {
    fn drop(&mut self) {
        unsafe { TfLiteInterpreterDelete(self.interpreter.as_ptr()) };
    }
}

unsafe impl<'a> Send for Interpreter<'a> {}
unsafe impl<'a> Sync for Interpreter<'a> {}

pub struct InputTensors<'i> {
    interpreter: &'i Interpreter<'i>,
    len: usize,
}

impl<'i> InputTensors<'i> {
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'i> std::ops::Index<usize> for InputTensors<'i> {
    type Output = Tensor;

    fn index(&self, index: usize) -> &Tensor {
        let index = i32::try_from(index).unwrap();
        unsafe { &*TfLiteInterpreterGetInputTensor(self.interpreter.interpreter.as_ptr(), index) }
    }
}

// TODO: impl iterator trait for InputTensors

pub struct InputTensorsMut<'i> {
    interpreter: &'i Interpreter<'i>,
    len: usize,
}

impl<'i> InputTensorsMut<'i> {
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'i> std::ops::Index<usize> for InputTensorsMut<'i> {
    type Output = Tensor;

    fn index(&self, index: usize) -> &Tensor {
        let index = i32::try_from(index).unwrap();
        unsafe { &*TfLiteInterpreterGetInputTensor(self.interpreter.interpreter.as_ptr(), index) }
    }
}

impl<'i> std::ops::IndexMut<usize> for InputTensorsMut<'i> {
    fn index_mut(&mut self, index: usize) -> &mut Tensor {
        let index = i32::try_from(index).unwrap();
        unsafe {
            &mut *TfLiteInterpreterGetInputTensor(self.interpreter.interpreter.as_ptr(), index)
        }
    }
}

pub struct OutputTensors<'i> {
    interpreter: &'i Interpreter<'i>,
    len: usize,
}

impl<'i> OutputTensors<'i> {
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'i> std::ops::Index<usize> for OutputTensors<'i> {
    type Output = Tensor;

    fn index(&self, index: usize) -> &Tensor {
        let index = i32::try_from(index).unwrap();
        unsafe { &*TfLiteInterpreterGetOutputTensor(self.interpreter.interpreter.as_ptr(), index) }
    }
}

// TODO: impl iterator trait for OutputTensors

impl Tensor {
    pub fn tensor_type(&self) -> TensorType {
        unsafe { TfLiteTensorType(self) }
    }
    pub fn num_dims(&self) -> usize {
        usize::try_from(unsafe { TfLiteTensorNumDims(self) }).unwrap()
    }
    pub fn dim(&self, i: usize) -> usize {
        assert!(i < self.num_dims());
        let i = i32::try_from(i).unwrap();
        usize::try_from(unsafe { TfLiteTensorDim(self, i) }).unwrap()
    }
    pub fn byte_size(&self) -> usize {
        unsafe { TfLiteTensorByteSize(self) }
    }

    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(TfLiteTensorName(self)) }
            .to_str()
            .unwrap()
    }

    pub fn shape(&self) -> Vec<usize> {
        let num_dims = self.num_dims();
        let mut dims = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            dims.push(self.dim(i));
        }
        dims
    }

    pub fn volume(&self) -> usize {
        self.shape().iter().fold(1, |acc, x| acc * x)
    }

    pub fn maprw<'a, T>(&'a mut self) -> Result<&'a mut [T], String> {
        let volume = self.volume();
        if std::mem::size_of::<T>() * volume > self.byte_size() {
            return Err(format!(
                "Tensor too small to map as {}",
                std::any::type_name::<T>()
            ));
        }
        let ptr = unsafe { TfLiteTensorData(self) } as *mut T;
        let volume = self.shape().iter().fold(1, |acc, x| acc * x);
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, volume as usize) })
    }

    pub fn mapro<'a, T>(&'a self) -> Result<&'a [T], String> {
        let volume = self.volume();
        if std::mem::size_of::<T>() * volume > self.byte_size() {
            return Err(format!(
                "Tensor too small to map as {}",
                std::any::type_name::<T>()
            ));
        }
        let ptr = unsafe { TfLiteTensorData(self) } as *mut T;
        Ok(unsafe { std::slice::from_raw_parts(ptr, volume as usize) })
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims = self.num_dims();
        let mut first = true;
        write!(f, "{}: ", self.name())?;
        for i in 0..dims {
            if !first {
                f.write_str("x")?;
            }
            first = false;
            write!(f, "{}", self.dim(i))?;
        }
        write!(f, " {:?}", self.tensor_type())
    }
}

impl std::fmt::Debug for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            unsafe { CStr::from_ptr(TfLiteTypeGetName(*self)) }
                .to_str()
                .unwrap(),
        )
    }
}

#[allow(dead_code)]
pub struct Model {
    ptr: ptr::NonNull<TfLiteModel>,
    model_mem: Vec<u8>,
}

impl Model {
    pub fn from_mem(model: Vec<u8>) -> Result<Self, String> {
        let m = unsafe { TfLiteModelCreate(model.as_ptr(), model.len()) };
        Ok(Model {
            ptr: ptr::NonNull::new(m).ok_or("TfLiteModelCreate returned NULL")?,
            model_mem: model,
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { TfLiteModelDelete(self.ptr.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    pub static MODEL: &'static [u8] =
        include_bytes!("testdata/ssd_mobilenet_v1_coco_2018_01_28.tflite");

    #[test]
    fn create_drop_model() {
        let _m = super::Model::from_mem(MODEL.to_vec()).unwrap();
    }

    #[test]
    fn lifecycle() {
        let m = super::Model::from_mem(MODEL.to_vec()).unwrap();
        let builder = super::Interpreter::builder();
        let mut interpreter = builder.build(&m).unwrap();
        println!(
            "interpreter with {} inputs, {} outputs",
            interpreter.inputs().len(),
            interpreter.outputs().len()
        );
        let inputs = interpreter.inputs();
        for i in 0..inputs.len() {
            println!("input: {:?}", inputs[i]);
        }
        let outputs = interpreter.outputs();
        for i in 0..outputs.len() {
            println!("output: {:?}", outputs[i]);
        }
    }

    #[cfg(feature = "edgetpu")]
    #[test]
    fn lifecycle_edgetpu() {
        static EDGETPU_MODEL: &'static [u8] = include_bytes!("testdata/edgetpu.tflite");
        let m = super::Model::from_static(EDGETPU_MODEL).unwrap();
        let mut builder = super::Interpreter::builder();
        let devices = super::edgetpu::Devices::list();
        if devices.is_empty() {
            panic!("need an edge tpu installed to run edge tpu tests");
        }
        let delegate = devices[0].create_delegate().unwrap();
        builder.add_owned_delegate(delegate);
        let mut interpreter = builder.build(&m).unwrap();
        println!(
            "interpreter with {} inputs, {} outputs",
            interpreter.inputs().len(),
            interpreter.outputs().len()
        );
        let inputs = interpreter.inputs();
        for i in 0..inputs.len() {
            println!("input: {:?}", inputs[i]);
        }
        let outputs = interpreter.outputs();
        for i in 0..outputs.len() {
            println!("output: {:?}", outputs[i]);
        }
    }
}
