#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"
#include <soloud_c.h>

typedef enum
{
    SOURCE_MOD_PLUG,
    SOURCE_WAV,
    SOURCE_WAV_STREAM,
    SOURCE_SPEECH,
    SOURCE_SFXR,
    SOURCE_COUNT,
} AudioSourceType;

typedef struct
{
    int voiceHandle;
    AudioSourceType type;
    AudioSource* source;
    //void Wav_destroy(Wav * aWav);
    //Wav * Wav_create();
    //int Wav_load(Wav * aWav, const char * aFilename);
    //int Wav_loadMem(Wav * aWav, unsigned char * aMem, unsigned int aLength);
    //double Wav_getLength(Wav * aWav);
    //void Wav_setLooping(Wav * aWav, int aLoop);
    //void Wav_setFilter(Wav * aWav, unsigned int aFilterId, Filter * aFilter);
    //void Wav_stop(Wav * aWav);

} PAudio;

Soloud* soloud;
void setupAudio()
{
    if (soloud == NULL)
    {
        soloud = Soloud_create();
        Soloud_initEx(soloud, 
            SOLOUD_CLIP_ROUNDOFF | SOLOUD_ENABLE_VISUALIZATION, 
            SOLOUD_AUTO, 
            SOLOUD_AUTO, 
            SOLOUD_AUTO);
    }
}

void destroySoloud()
{
    if (soloud != NULL)
    {
        Soloud_deinit(soloud);
        Soloud_destroy(soloud);
    }
}

PAudio loadSpeech(const char* speech)
{
    PAudio newAudio = 
    {
        0,
        SOURCE_SPEECH,
        Speech_create(),
    };
    
    Speech_setText(newAudio.source, speech);
    return newAudio;
}

// Supports wav, ogg
// 669, abc, amf, ams, dbm, dmf, dsm, far, it, j2b, mdl, med, mid, mod, mt2, mtm, okt, pat, psm, ptm, s3m, stm, ult, umx, xm
PAudio loadAudio(const char* filename)
{
    PAudio newAudio = {0};

    {
        Modplug* source = Modplug_create();
        if (Modplug_load(source, filename) == 0)
        {
            newAudio.type = SOURCE_MOD_PLUG;
            newAudio.source = source;
            return newAudio;
        }
        Modplug_destroy(source);
    }

    {
        Wav* source = Wav_create();
        if (Wav_load(source, filename) == 0)
        {
            newAudio.type = SOURCE_WAV;
            newAudio.source = source;
            return newAudio;
        }
        Wav_destroy(source);
    }

    return newAudio;
}

void playAudio(PAudio audio)
{
    audio.voiceHandle = Soloud_play(soloud, audio.source);
}

void stopAudio(PAudio audio)
{
    switch (audio.type)
    {
    case SOURCE_MOD_PLUG:       Modplug_stop(audio.source); break;
    case SOURCE_WAV:            Wav_stop(audio.source); break;
    case SOURCE_WAV_STREAM:     WavStream_stop(audio.source); break;
    default:                    break;
    }
    audio.voiceHandle = 0;
}

void destroyAudio(PAudio audio)
{
    switch (audio.type)
    {
    case SOURCE_MOD_PLUG:       Modplug_destroy(audio.source); break;
    case SOURCE_WAV:            Wav_destroy(audio.source); break;
    case SOURCE_WAV_STREAM:     WavStream_destroy(audio.source); break;
    default:                    break;
    }
    audio.voiceHandle = 0;
}

void setAudioPause(PAudio audio, int pause)
{
    Soloud_setPause(soloud, audio.voiceHandle, pause);
}

void setAudioPan(PAudio audio, float pan)
{
    Soloud_setPan(soloud, audio.voiceHandle, pan);
}

void setAudioVolume(PAudio audio, float volume)
{
    Soloud_setVolume(soloud, audio.voiceHandle, volume);
}

int i = 0;
PAudio speech;

void setup()
{
    setupAudio(); // TODO:

    speech = loadSpeech("1 2 3       A B C        Doooooo    Reeeeee    Miiiiii    Faaaaaa    Soooooo    Laaaaaa    Tiiiiii    Doooooo!");
    playAudio(speech);

    printf("Playing..\n");
}

void draw()
{
    background(gray(122));
    while (Soloud_getActiveVoiceCount(soloud) > 0)
    {
        float * v = Soloud_calcFFT(soloud);
        int p = (int)(v[10] * 30);
        if (p > 59) p = 59;
        for (i = 0; i < p; i++)
            printf("=");
        for (i = p; i < 60; i++)
            printf(" ");
        printf("\r");
        printf("%c\r", "|\\-/"[i&3]);
        i++;
    }
}

void shutdown()
{
    printf("\nFinished.\n");

    destroyAudio(speech);
    destroySoloud(soloud);

    printf("Cleanup done.\n");
}
