#!/usr/bin/env bash

# Rather than deal with the arcane mysticism of bash arrays
# and loops, I'm doing the lazy thing. 
# Apparently spaces inside elements are problematic, and there's an issue
# with scp not copying a file - when passed as an arg - that
# works fine when explicit
scp -r ~/source/ImageCaptionLearn/src/ clgrad3:~/source/ImageCaptionLearn/staging_0/ >/dev/null
scp ~/source/ImageCaptionLearn/pom.xml clgrad3:~/source/ImageCaptionLearn/staging_0/ >/dev/null
scp ~/source/ImageCaptionLearn/build.sh clgrad3:~/source/ImageCaptionLearn/staging_0/ >/dev/null
scp ~/source/ImageCaptionLearn/README clgrad3:~/source/ImageCaptionLearn/staging_0/ >/dev/null

scp -r ~/source/ImageCaptionLearn/src/ clgrad3:~/source/ImageCaptionLearn/staging_1/ >/dev/null
scp ~/source/ImageCaptionLearn/pom.xml clgrad3:~/source/ImageCaptionLearn/staging_1/ >/dev/null
scp ~/source/ImageCaptionLearn/build.sh clgrad3:~/source/ImageCaptionLearn/staging_1/ >/dev/null
scp ~/source/ImageCaptionLearn/README clgrad3:~/source/ImageCaptionLearn/staging_1/ >/dev/null

scp -r ~/source/ImageCaptionLearn/src/ clgrad3:~/source/ImageCaptionLearn/staging_2/ >/dev/null
scp ~/source/ImageCaptionLearn/pom.xml clgrad3:~/source/ImageCaptionLearn/staging_2/ >/dev/null
scp ~/source/ImageCaptionLearn/build.sh clgrad3:~/source/ImageCaptionLearn/staging_2/ >/dev/null
scp ~/source/ImageCaptionLearn/README clgrad3:~/source/ImageCaptionLearn/staging_2/ >/dev/null



#args=('~/source/ImageCaptionLearn/pom.xml' '~/source/ImageCaptionLearn/build.sh' '~/source/ImageCaptionLearn/README')
#dests=('staging_0/' 'staging_1/' 'staging_2/')

#for dest in "$dests"; do
#    scp -r ~/source/ImageCaptionLearn/src/ "clgrad2:~/source/ImageCaptionLearn/$dest" >/dev/null
#    for arg in "$args"; do
#        scp "$arg" clgrad2:~/source/ImageCaptionLearn/"$dest" >/dev/null
#    done
#done
