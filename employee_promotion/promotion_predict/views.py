from django.contrib.auth.hashers import make_password, BCryptSHA256PasswordHasher
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import CustomAuthenticationForm, CustomUserCreationForm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64

def index(request):
    return render(request, "index.html")

def login_view(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('index/')
    else:
        form = CustomAuthenticationForm()
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')

def create_account_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            password = form.cleaned_data['password1']
            
            # Use a different password hasher (e.g., bcrypt)
            hasher = BCryptSHA256PasswordHasher()
            hashed_password = make_password(password, hasher=hasher)

            user = form.save(commit=False)
            user.password = hashed_password
            user.save()

            login(request, user)
            return redirect('index/')
        else:
            print("Form is not valid. Errors:", form.errors)
    else:
        form = CustomUserCreationForm()
    return render(request, 'create_account.html', {'form': form})

def result(request):
    X_train = pd.read_csv(r'X_train.csv').head(2000)
    X_train = X_train.drop(['is_promoted'], axis=1)
    y_train = pd.read_csv(r'y_train.csv').head(2000)

    def pad(data):
        block_size = algorithms.AES.block_size // 8
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding

    def encrypt_data(data, key):
        if isinstance(data, str):
            data = data.encode('utf-8')
        cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(pad(data)) + encryptor.finalize()
        return base64.b64encode(encrypted_data).decode("utf-8")

    # Set your encryption key (16 bytes for AES-128)
    key = 'b6fa11f0f1aad4dd123b03ca87fa3ec1'
    key = bytes.fromhex(key)

    for col in X_train.columns:
        X_train[col] = X_train[col].astype(str).apply(lambda x: encrypt_data(x.encode('utf-8'), key))  
    def decode_and_convert_to_int(byte_string):
        decoded_bytes = base64.b64decode(byte_string)
        result_integer = int.from_bytes(decoded_bytes, byteorder='big')
        return result_integer

    # Apply the function to all elements in the DataFrame
    X_train = X_train.applymap(decode_and_convert_to_int)
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
    RF_model = RandomForestClassifier()
    # Train the model
    RF_model = RF_model.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    val9 = float(request.GET['n9'])
    val10 = float(request.GET['n10'])
    val11 = float(request.GET['n11'])
    val12 = float(request.GET['n12'])

    df = pd.DataFrame([[val1, val2, val3, 
                           val4, val5, val6, val7, val8, val9, val10, val11, val12]], columns=X_train.columns)
    for col in df.columns:
        df[col] = df[col].astype(str).apply(lambda x: encrypt_data(x.encode('utf-8'), key)) 

    df = df.applymap(decode_and_convert_to_int)
    pred = RF_model.predict(df)
 
    result1 = ""
    if pred == [1]:
        result1 = "You might get promoted"
    else:
        result1 = "You have no chance of getting promoted"
 
    return render(request, "index.html", {"result2": result1})
