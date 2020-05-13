#include "mbed.h"
#include <cmath>
#include "DA7212.h"
#include "uLCD_4DGL.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

DA7212 audio;

Serial pc(USBTX, USBRX);
InterruptIn button(SW2);
InterruptIn keyboard0(SW3);
DigitalOut green_led(LED2);
uLCD_4DGL uLCD(D1, D0, D2);

EventQueue queue(32 * EVENTS_EVENT_SIZE);
EventQueue queue2(32 * EVENTS_EVENT_SIZE);
Thread t;
Thread t2(osPriorityNormal,120*1024/*120K stack size*/);

Timer timer;
int score = 0;
int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[42];
int serialCount = 0;

int beats[10] = {1,0,2,0,0,1,1,0,2,2};

int song[3] = {0,261,392};

int song1[42] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261};

int noteLength[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};

void playNote2(){
      for(int k = 0; k < 42; k++){
        int length = noteLength[k];
        while(length--)
        {
          for (int i = 0; i < kAudioTxBufferSize; i++)
          {
            waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / song1[k])) * ((1<<16) - 1));
          }
          for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
          {
            audio.spk.play(waveform, kAudioTxBufferSize);
          }
          if(length <= 1) wait(1.0);
        }
      }
      audio.spk.pause();
}


int PredictGesture(float* output) {

  // How many times the most recent gesture has been matched in a row

  static int continuous_count = 0;

  // The result of the last prediction

  static int last_predict = -1;


  // Find whichever output has a probability > 0.8 (they sum to 1)

  int this_predict = -1;

  for (int i = 0; i < label_num; i++) {

    if (output[i] > 0.8) this_predict = i;

  }

  // No gesture was detected above the threshold

  if (this_predict == -1) {

    continuous_count = 0;

    last_predict = label_num;

    return label_num;

  }


  if (last_predict == this_predict) {

    continuous_count += 1;

  } else {

    continuous_count = 0;

  }

  last_predict = this_predict;


  // If we haven't yet had enough consecutive matches for this gesture,

  // report a negative result

  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {

    return label_num;

  }

  // Otherwise, we've seen a positive result, so clear all our variables

  // and report it

  continuous_count = 0;

  last_predict = -1;


  return this_predict;

}

void playNote(){
  uLCD.cls();
  uLCD.locate(1,2);
  uLCD.printf("102001122");
  for(int k = 0; k < sizeof(beats)/sizeof(beats[0]); k++){
      int index = beats[k];
      for (int i = 0; i < kAudioTxBufferSize; i++)
      {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / song[index])) * ((1<<16) - 1));
      }
      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
      {
        audio.spk.play(waveform, kAudioTxBufferSize);
      }
      wait(1);
      audio.spk.pause();
  }
  audio.spk.pause();
  uLCD.cls();
  uLCD.locate(1,2);
  uLCD.printf("Your Score = %d",score);

}

void DNN_select(){
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  bool should_clear_buffer = false;
  bool got_data = false;
  int gesture_index;
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    // return -1;
  }
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  TfLiteTensor* model_input = interpreter->input(0);

  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    // return -1;

  }
  int input_length = model_input->bytes / sizeof(float);
  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    // return -1;
  }
  error_reporter->Report("Set up successful...\n");
  while(true){
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                input_length, should_clear_buffer);
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }
    gesture_index = PredictGesture(interpreter->output(0)->data.f);
    should_clear_buffer = gesture_index < label_num;

    
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
      int target = int(timer.read());
      if(gesture_index == 0 && beats[target] == 1){
        score ++;
      }
      else if(gesture_index == 2 && beats[target] == 2){
        score ++;
      }
    }
  }
}

void triggering(){
  timer.start();
  score = 0;

  queue.call(playNote2);
  // queue.call(playNote);
  //queue2.call(DNN_select);
}

int main(void)
{
  green_led = 1;
  t.start(callback(&queue, &EventQueue::dispatch_forever));
  t2.start(callback(&queue2, &EventQueue::dispatch_forever));

  uLCD.reset();
  uLCD.locate(1,2);
  uLCD.printf("102001122");
  keyboard0.rise(triggering);
  
}