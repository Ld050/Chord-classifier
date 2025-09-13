# Auto-generated from notebooks/Chord_classifier.ipynb
# Cells are concatenated in order.

# %% Cell 1
import pandas as pd
import numpy as np

freq_0 = 27.5
freq = []

for i in range(108):
    freq.append(freq_0)
    freq_0 *= 2**(1/12)


freq = np.reshape(np.round(freq,2), (9, 12))
cols = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
df_note_freqs = pd.DataFrame(freq, columns=cols)
print("Частоты нот (Гц)")
df_note_freqs.head(10)

# %% Cell 2
!unzip majmin_data.zip

# %% Cell 3
import IPython

chord_1_maj = "/content/Audio_Files/Major/Major_0.wav"

chord_1_min = "/content/Audio_Files/Minor/Minor_0.wav"

print("Аккорд C (До) мажор")
IPython.display.Audio(chord_1_maj, rate = 44100)

# %% Cell 4
print("Аккорд C (До) минор")
IPython.display.Audio(chord_1_min, rate = 44100)

# %% Cell 5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, find_peaks

# %% Cell 6
path = "/content/Audio_Files/Major/Major_0.wav"
fs, signal = wavfile.read(path)
N = len(signal)
time = np.linspace(0., N/fs, N)


y_freq = fftfreq(N, 1/fs)[:N//2]
signal_f = fft(signal)
signal_abs = 2.0/N * np.abs(signal_f[0:N//2])


plt.plot(time, signal)
plt.title("Сигнал во временной области")
plt.xlabel('Время, с')
plt.show()

# %% Cell 7
plt.plot(y_freq, signal_f_onesided)
plt.title("Сигнал в частотной области")
plt.xlabel('Частота, Гц')
plt.show()

# %% Cell 8
plt.plot(time[(N//2):(N//2+480)], signal[(N//2):(N//2+480)])
plt.title("Сигнал во временной области (Прибл.)")
plt.xlabel('Время, с')
plt.show()

# %% Cell 9
plt.plot(y_freq[:5000], signal_f_onesided[:5000])
plt.title("Сигнал в частотной области (Прибл.)")
plt.xlabel('Частота, Гц')
fig.tight_layout()
plt.show()

# %% Cell 10
threshold = signal_abs.max()*5/100
peaks, _ = find_peaks(signal_abs, distance=10, height = threshold)

base_50 = np.abs(y_freq - 50).argmin()
peaks = peaks[peaks>base_50]
harmonics = y_freq[peaks]
print("Пики гармоник: {}".format(np.round(harmonics)))


i = peaks.max() + 100
plt.plot(y_freq[:i], signal_abs[:i], color = 'black')
plt.plot(y_freq[peaks], signal_abs[peaks], "x", color = 'red')
plt.xlabel('Частота, Гц')
plt.show()

# %% Cell 11
def find_harmonics(path, print_peaks=False):
    fs, X = wavfile.read(path)
    N = len(X)
    X_F = fft(X)
    X_F_abs = 2.0/N * np.abs(X_F[0:N//2])
    freqs = fftfreq(N, 1/fs)[:N//2]
    base_50 = np.abs(freqs - 50).argmin()

    h = X_F_abs.max()*5/100
    peaks, _ = find_peaks(X_F_abs, distance=10, height = h)
    peaks = peaks[peaks>base_50]
    harmonics = np.round(freqs[peaks])

    if print_peaks:
        i = peaks.max() + 100
        plt.plot(freqs[:i], X_F_abs[:i], color = 'black')
        plt.plot(freqs[peaks], X_F_abs[peaks], "x", color = 'red')
        plt.xlabel('Частота, Гц')
        plt.show()
    return harmonics

# %% Cell 12
path_example = "/content/Audio_Files/Major/Major_1.wav"

find_harmonics(path_example, print_peaks=True)

# %% Cell 13
import os

path = "/content/Audio_Files"
data = []

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        foldername = os.path.basename(dirname)
        full_path = os.path.join(dirname, filename)
        freq_peaks = find_harmonics(full_path)

        cur_data = [foldername, filename, freq_peaks.min(), freq_peaks.max(), len(freq_peaks)]
        cur_data.extend(freq_peaks)

        data.append(cur_data)
max_harm_length = max(len(row[5:]) for row in data)
cols = ["Chord Type", "File Name", "Min Harmonic", "Max Harmonic", "i of Harmonics"] + \
       ["Harmonic {}".format(i+1) for i in range(max_harm_length)]
df = pd.DataFrame(data, columns=cols)
df

# %% Cell 14
miss_values = df.isnull().sum().sort_values(ascending=False)
miss_values[miss_values>0]

# %% Cell 15
df["Interval 1"] = df["Harmonic 2"].div(df["Harmonic 1"], axis=0)

fig, axes = plt.subplots(2, 1, figsize=(7, 5))
sns.kdeplot(ax=axes[0], data=df, x="Harmonic 2", hue="Chord Type", fill=True)
sns.kdeplot(ax=axes[1], data=df, x="Interval 1", hue="Chord Type", fill=True)
fig.tight_layout()
plt.show()

# %% Cell 16
for i in range(1,21):
    inter_c = "Interval {}".format(i)
    harm_c = "Harmonic {}".format(i+1)
    harm_p = "Harmonic {}".format(i)
    df[inter_c] = df[harm_c].div(df[harm_p], axis=0)

df.head()

# %% Cell 17
fig, axes = plt.subplots(5, 4, figsize=(12, 8))
for i in range(1,21):
    plt.subplot(5, 4, i)
    sns.kdeplot(data=df, x="Interval {}".format(i), hue="Chord Type", fill=True)
fig.tight_layout()
plt.show()

# %% Cell 18
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# %% Cell 19
columns = ["Interval 1", "Interval 2", "Interval 3", "Interval 4"]
X_train, x_test, y_train, y_test = train_test_split(df[columns], df["Chord Type"], test_size=0.4, random_state=0)
dec_sc = cross_val_score(DecisionTreeClassifier(random_state=0), X_train, y_train, cv=10).mean()
for_sc = cross_val_score(RandomForestClassifier(random_state=0), X_train, y_train, cv=10).mean()
print("Decision score: " , dec_sc)
print("Forest score: ", for_sc)

# %% Cell 20
classifier = RandomForestClassifier(random_state=0)

classifier.fit(train_X, train_y)
pred_y = classifier.predict(val_X)


print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_y))
print("Accuracy Score: {:.2f}".format(accuracy_score(y_test, pred_y)))
