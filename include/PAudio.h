#ifndef PAUDIO_H
#define PAUDIO_H

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

extern Soloud* soloud;

void setupPAudio();
void teardownPAudio();

PAudio loadSpeech(const char* speech);
PAudio loadAudio(const char* filename);
void playAudio(PAudio audio);
void stopAudio(PAudio audio);
void destroyAudio(PAudio audio);
void setAudioPause(PAudio audio, int pause);
void setAudioPan(PAudio audio, float pan);
void setAudioVolume(PAudio audio, float volume);

#endif // PAUDIO_H

#ifdef PAUDIO_IMPLEMENTATION

Soloud* soloud;

void setupPAudio()
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

void teardownPAudio()
{
    if (soloud != NULL)
    {
        Soloud_deinit(soloud);
        Soloud_destroy(soloud);

        soloud = NULL;
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
        Wav* source = Wav_create();
        if (Wav_load(source, filename) == 0)
        {
            newAudio.type = SOURCE_WAV;
            newAudio.source = source;
            return newAudio;
        }
        Wav_destroy(source);
    }
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

    return newAudio;
}

void playAudio(PAudio audio)
{
    if (audio.source == NULL) return;

    audio.voiceHandle = Soloud_play(soloud, audio.source);
}

void stopAudio(PAudio audio)
{
    if (audio.voiceHandle == 0) return;

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
    if (audio.voiceHandle == 0) return;

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

#endif // PAUDIO_IMPLEMENTATION