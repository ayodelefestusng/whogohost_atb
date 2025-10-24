from email.policy import default
from pyexpat import model
from django.db import models

# Create your models here.
class Conversation(models.Model):
    # Removed user ForeignKey. Using session_id instead.
    conversation_id = models.CharField(max_length=255, unique=True, db_index=True) # Unique per session for active conversations
    started_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True) 
    summary = models.TextField(null=True, blank=True)  # Added field type and allowed blank
    
    message_count = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Conversation {self.conversation_id} (Session: {self.conversation_id})"
    

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    text = models.TextField()
    is_user = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)
    attachment = models.FileField(upload_to='chat_attachments/', null=True, blank=True)

    class Meta:
        ordering = ['timestamp']

class Tenant(models.Model):
    tenant_id = models.IntegerField(primary_key=True)
    tenant_name = models.CharField(max_length=255)
    tenant_kss = models.FileField(upload_to='profiles/', blank=True, null=True)
    chatbot_greeting = models.TextField()
    agent_node_prompt = models.TextField()
    final_answer_prompt = models.TextField()
    summary_prompt = models.TextField()
    is_active=models.BooleanField(default=False)
    tenant_description = models.CharField(max_length=255,blank=True,null=True)

    def __str__(self):
        return self.tenant_name