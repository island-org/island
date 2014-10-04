#define SKETCH_2D_IMPLEMENTATION
#include "sketch2d.h"
#include <soloud_c.h>

int i = 0;
Soloud* soloud;
Speech* speech;

void setup()
{
    soloud = Soloud_create();
    speech = Speech_create();

    Speech_setText(speech, "1 2 3       A B C        Doooooo    Reeeeee    Miiiiii    Faaaaaa    Soooooo    Laaaaaa    Tiiiiii    Doooooo!");

    Soloud_initEx(soloud,SOLOUD_CLIP_ROUNDOFF | SOLOUD_ENABLE_VISUALIZATION, SOLOUD_AUTO, SOLOUD_AUTO, SOLOUD_AUTO);

    Soloud_setGlobalVolume(soloud, 4);
    Soloud_play(soloud, speech);

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

    Soloud_deinit(soloud);

    Speech_destroy(speech);
    Soloud_destroy(soloud);

    printf("Cleanup done.\n");
}
