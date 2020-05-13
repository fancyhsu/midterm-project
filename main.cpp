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

Timer timer;
EventQueue queue(32 * EVENTS_EVENT_SIZE);
EventQueue queue2(32 * EVENTS_EVENT_SIZE);
Thread t;
Thread t2(osPriorityNormal,120*1024/*120K stack size*/);

int idC = 0, songnum = 0, status = 0, mode = 0, score = 0;
int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[42];
int serialCount = 0;

float song[3][42]={
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  428, 428, 392, 392, 440, 440, 392,
  500, 500, 330, 330, 258, 258, 428,
  392, 392, 500, 500, 330, 330, 258,
  392, 392, 500, 500, 330, 330, 258,
  428, 428, 392, 392, 440, 440, 392,
  500, 500, 330, 330, 258, 258, 428,
  350, 350, 277, 277, 440, 440, 277,
  349, 349, 488, 488, 294, 294, 350,
  277, 277, 349, 349, 488, 488, 294,
  277, 277, 349, 349, 488, 488, 294,
  350, 350, 277, 277, 440, 440, 392,
  349, 349, 488, 488, 294, 294, 261};

int noteLength[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};

int beats[10] = {1,0,2,0,0,1,1,0,2,2};

int PredictGesture(float* output) {
  static int continuous_count = 0;
  static int last_predict = -1;
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }
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
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  continuous_count = 0;
  last_predict = -1;
  return this_predict;
}

void loadSignal(void){
  green_led = 0;
  int i = 0;
  serialCount = 0;
  while(i < 42)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      
      serialCount++;
      if(serialCount == 7)
      {
        serialInBuffer[serialCount] = '\0';
        song[i/42][i%42] = (float) atof(serialInBuffer);
        serialCount = 0;
        i++;
        printf("%f\r\n",(float) atof(serialInBuffer));
      }
    }
  }
  green_led = 1;
}

void playNote2(){
      for(int k = 0; k < 42; k++){
        if(timer.read()>10) break;
        int length = noteLength[k];
        while(length--)
        {
          for (int i = 0; i < kAudioTxBufferSize; i++)
          {
            waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / song[0][k])) * ((1<<16) - 1));
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

void playNote(){
      for(int k = 0; k < 42; k++){
        
        int length = noteLength[k];
        while(length--)
        {
          for (int i = 0; i < kAudioTxBufferSize; i++)
          {
            waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / song[songnum][k])) * ((1<<16) - 1));
          }
          for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
          {
            audio.spk.play(waveform, kAudioTxBufferSize);
          }
          if(length <= 1) wait(1.0);
          if(status != 1) break;
        }
        if(status != 1) break;
      }
      audio.spk.pause();
      audio.spk.pause();
      uLCD.cls();
      uLCD.locate(1,2);
      uLCD.printf("Your Score = %d",score);
}

void loadSignalHandler(void){
  queue.call(loadSignal);
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
      if(status == 2) {
        
        mode = (mode+1)%3;
        uLCD.cls();
        uLCD.locate(1,2);
        if(mode == 0) uLCD.printf("Mode : Play next song");
        else if(mode == 1)  uLCD.printf("Mode : Play previous song");
        else if(mode == 2)  uLCD.printf("Mode : Pick song");
        else if(mode == 3)  uLCD.printf("Mode : Taiko");
      }
      else if(status == 3) {
        songnum = (songnum+1)%3;
        uLCD.cls();
        uLCD.locate(1,2);
        if(songnum == 0) uLCD.printf("Song 1");
        else if(songnum == 1)  uLCD.printf("Song 2");
        else if(songnum == 2)  uLCD.printf("Song 3");
      }
      else if(status == 4) {
        int target = int(timer.read());
        if(gesture_index == 0 && beats[target] == 1){
          score ++;
        }
        else if(gesture_index == 2 && beats[target] == 2){
          score ++;
        }
      }
    }
    if(status == 1) playNote();
  }
}

void change_status(void) {
  
  uLCD.cls();
  uLCD.locate(1,2);
  if(status == 0){
    status = 1;
    mode = 0;
    uLCD.printf("Playing song %d now",songnum+1);
  }
  else if(status == 1){
    uLCD.printf("Mode : Play next song");
    status = 2;
  }
  else if(status == 2){
    if(mode == 0 || mode == 1){
      uLCD.printf("Push button to play!");
      if(mode == 0) songnum = (songnum+1)%3;
      else if(mode == 1) songnum = (songnum+2)%3;
      status = 0;
    }
    else if(mode == 2){
      uLCD.printf("Song %d",songnum+1);
      status = 3;
    }
    else if(mode == 3){
      uLCD.printf("Taiko : 1020011022");
      timer.reset();
      timer.start();
      queue.call(playNote2);
      status = 4;
    }
  }
  else if(status == 3){
    uLCD.printf("Push button to play!");
    status = 0;
  }
  else if(status == 4){
    status = 2;
    uLCD.printf("Mode : Play next song");
  }
}
void triggering(){
  queue2.call(DNN_select);
}

int main(void)
{
  green_led = 1;
  t.start(callback(&queue, &EventQueue::dispatch_forever));
  t2.start(callback(&queue2, &EventQueue::dispatch_forever));

  loadSignal();
  uLCD.reset();
  uLCD.locate(1,2);
  uLCD.printf("Push button to play!");
  keyboard0.rise(triggering);
  button.rise(queue.event(change_status));
  
}