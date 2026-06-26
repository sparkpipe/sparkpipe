#include <stdio.h>
#include <stdlib.h>

int main(int argument_count, char **arguments)
{
    FILE *object_file;
    FILE *counter_file;
    unsigned long validation_count;

    if (argument_count < 3)
    {
        return 2;
    }
    object_file = fopen(arguments[argument_count - 1], "rb");
    if (object_file == 0)
    {
        return 3;
    }
    if (fgetc(object_file) == EOF)
    {
        fclose(object_file);
        return 4;
    }
    fclose(object_file);

    validation_count = 0u;
    counter_file = fopen(arguments[1], "r");
    if (counter_file != 0)
    {
        if (fscanf(counter_file, "%lu", &validation_count) != 1)
        {
            fclose(counter_file);
            return 5;
        }
        fclose(counter_file);
    }
    counter_file = fopen(arguments[1], "w");
    if (counter_file == 0)
    {
        return 6;
    }
    fprintf(counter_file, "%lu\n", validation_count + 1u);
    if (fclose(counter_file) != 0)
    {
        return 7;
    }
    return 0;
}
