
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from theano import config

from Confs import default

# Define font style (Hand Written) and color.
FONT_NAME = default('fontName')
FONT_SIZE = int(default('fontSize'))

MOTIFS_STRING = default('motifs')

MOTIF_HEIGHT = int(default('docHeight'))
MOTIF_LENGTH = int(default('filtLength'))

# Define the parameters
NB_OCCURENCE_PER_DOC = int(default('nbOccMotifPerDoc'))
DOC_LENGTH = int(default('docLength'))
NB_OBSERVATION_PER_OCC_MIN = int(default('nbObsPerMotifMin'))
NB_OBSERVATION_PER_OCC_MAX = int(default('nbObsPerMotifMax'))
NB_TDOC = int(default('nbGeneratedTdoc'))
DEF_PERC_NOISE = default('percNoise')
PERC_NOISE = float(DEF_PERC_NOISE)

# Files parameters
DOC_PREFIX = default('prefDoc')+"noise"+str(DEF_PERC_NOISE)+"_"
DOC_EXT = default('extDoc')
MOTIF_PREFIX = default('prefMotif')
IMG_EXT = default('extImg')

def get_size(string, font):
    '''
    Returns the given text size in pixels
    '''
    test_img = Image.new('L', (1, 1))
    test_draw = ImageDraw.Draw(test_img)
    return test_draw.textsize(string, font)

# Convert String to Matrix
def string_to_matrix(string):
    '''
    Returns the pixel matrix corresponding to the given string
    '''
    # Define the Text Color and the Background
    color_text = "White"
    color_background = "Black"

    #Define the image font and resize the nword in a rectangle that suit it
    font = ImageFont.truetype(FONT_NAME, FONT_SIZE)
    str_l, str_h = get_size(string, font)
    pos_l = max(1, (MOTIF_LENGTH-str_l)//2)
    pos_h = max(1, (MOTIF_HEIGHT-str_h)//2)
    img = Image.new('L', (MOTIF_LENGTH, MOTIF_HEIGHT), color_background)
    drawing = ImageDraw.Draw(img)
    drawing.text((pos_l, pos_h), string, fill=color_text, font=font)

    path_motif = MOTIF_PREFIX+string+IMG_EXT
    img.save(path_motif)

    motif = np.asarray(img, config.floatX)   # Motif Matrix

    return motif


def motif_matrix_to_norm_vector(motif):
    '''
    Converts Motif to vector in form of Distribution.
    '''
    motif_as_vector = motif.reshape((MOTIF_LENGTH*MOTIF_HEIGHT))
    # normalizing as a distribution
    motif_as_vector = motif_as_vector / motif_as_vector.sum()
    motif_as_vector *= .999999

    return motif_as_vector

def creat_tdoc_from_motif():
    '''
    Returns a matrix representing a tdoc
    '''
    tdoc = np.zeros((DOC_LENGTH, MOTIF_HEIGHT), dtype=np.uint8)

    for _ in range(NB_OCCURENCE_PER_DOC):
        # Random-motif's weights that we will draw from
        index_motif = np.random.randint(len(MOTIFS_AS_NORM_VECTOR))
        motif = MOTIFS_AS_NORM_VECTOR[index_motif]

        start_time = np.random.randint(DOC_LENGTH - MOTIF_LENGTH)
        nb_obs = np.random.randint(NB_OBSERVATION_PER_OCC_MAX - NB_OBSERVATION_PER_OCC_MIN)
        nb_obs += NB_OBSERVATION_PER_OCC_MIN
        nb_obs *= MOTIFS_STRING[index_motif][1]

        for _ in range(nb_obs):
            # Draw samples from a multinomial distribution.
            pixel_position = np.argmax(np.random.multinomial(1, motif))

            h_pos = pixel_position // MOTIF_LENGTH
            t_pos = pixel_position % MOTIF_LENGTH

            time = t_pos + start_time
            tdoc[time][h_pos] += 1

    noise_intensity = (PERC_NOISE
                       *((NB_OBSERVATION_PER_OCC_MAX+NB_OBSERVATION_PER_OCC_MIN)//2)
                       *NB_OCCURENCE_PER_DOC)
    noise_intensity_max = 0.1*tdoc.max()
    while noise_intensity > 0:
        t_noise = np.random.randint(DOC_LENGTH)
        w_noise = np.random.randint(MOTIF_HEIGHT)

        v_noise = np.random.randint(int(noise_intensity_max))

        tdoc[t_noise][w_noise] += v_noise
        noise_intensity -= v_noise
    return tdoc



# See all motifs matrix in one big matrix like [[],[],[],[],[]]
MOTIFS_AS_MATRIX = [string_to_matrix(motif[0]) for motif in MOTIFS_STRING]

# Convert the big normalized matrix into a vector
MOTIFS_AS_NORM_VECTOR = [motif_matrix_to_norm_vector(motif) for motif in MOTIFS_AS_MATRIX]

# Generate documents
if __name__ == '__main__':
    for i in range(0, NB_TDOC):
        temp_doc = creat_tdoc_from_motif()

        path = DOC_PREFIX+str((i+1))

        plt.imshow(np.transpose(temp_doc), cmap=cm.Greys_r)
        plt.savefig(path+IMG_EXT)

        l = []
        output = open(path+DOC_EXT, 'w')
        for length in range(DOC_LENGTH):
            for height in range(MOTIF_HEIGHT):
                if temp_doc[length][height] != 0.0:
                    output.write("%s:%s " %(height, temp_doc[length][height]))
            output.write("\n")
        output.close()
