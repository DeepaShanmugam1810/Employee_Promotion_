# authentication/forms.py
from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from .models import CustomUser

class CustomAuthenticationForm(AuthenticationForm):
    username = forms.CharField(max_length=254, widget=forms.TextInput(attrs={'autofocus': True}))

    class Meta:
        model = CustomUser
        fields = ['username', 'password']

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'password1']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the password2 field
        del self.fields['password2']
