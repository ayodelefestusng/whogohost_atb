from django.contrib import admin
from .models import Tenant,Conversation,Message

# Register your models here.
admin.site.register(Tenant) 
admin.site.register(Conversation) 

admin.site.register(Message) 

