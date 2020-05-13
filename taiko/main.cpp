#include "mbed.h"
#include <cmath>
#include "DA7212.h"
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
EventQueue queue(32 * EVENTS_EVENT_SIZE);
Thread t;

int idC = 0, song = 0, status = 0, mode = 0;
int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[42];
int serialCount = 0;

DigitalOut green_led(LED2);

int song[42] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261};
int song2[42] = {
  428, 428, 392, 392, 440, 440, 392,
  500, 500, 330, 330, 258, 258, 428,
  392, 392, 500, 500, 330, 330, 258,
  392, 392, 500, 500, 330, 330, 258,
  428, 428, 392, 392, 440, 440, 392,
  500, 500, 330, 330, 258, 258, 428};
int song3[42] = {
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

void loadSignal(void){
  green_led = 0;
  serialCount = 0;
  audio.spk.pause();

  while(serialCount < 42)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
    }
  }
  green_led = 1;
}

void playNote(int freq){
  for(int i = 0; i < 42; i++){
    int length = noteLength[i];
    while(length--)
    {
      for (int i = 0; i < kAudioTxBufferSize; i++)
      {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / song2[i])) * ((1<<16) - 1));
      }
      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
      {
        audio.spk.play(waveform, kAudioTxBufferSize);
      }
      if(length <= 1) wait(1.0);
    }
  }
  
}

void loadSignalHandler(void) {queue.call(loadSignal);}

void mode_select(){

}

void song_select(){
  
}

void playNoteC(void) {
  if(status == 0){
    idC = queue.call(playNote);
    status = 1;
  }
  else if(status == 1){
    queue.cancel(idC);
    mode = 1;
    idC = queue.call(mode_select);
    status = 2;
  }
  else if(status == 2){
    queue.cancel(idC);
    if(mode == 0 || mode == 1)
      status == 0;
    else if(mode == 2){
      idC = queue.call(song_select);
      status == 3;
    }
  }
  else if(status == 3){
    queue.cancel(idC);
    status = 0;
  }
  // for(int i = 0; i < 42; i++){
  //   int length = noteLength[i];
  //   while(length--)
  //   {
  //     playNote(song2[i]);
  //     if(length <= 1) wait(1.0);
  //   }
  // }
}

int main(void)
{
  green_led = 1;
  t.start(callback(&queue, &EventQueue::dispatch_forever));
  button.rise(queue.event(loadSignalHandler));
  
  keyboard0.rise(queue.event(playNoteC));

}