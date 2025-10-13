from django.db import models

# Create your models here.
class Conversation(models.Model):
    # Removed user ForeignKey. Using session_id instead.
    session_id = models.CharField(max_length=255, unique=True, db_index=True) # Unique per session for active conversations
    started_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Conversation {self.session_id} (Session: {self.session_id})"
    

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    text = models.TextField()
    is_user = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)
    attachment = models.FileField(upload_to='chat_attachments/', null=True, blank=True)

    class Meta:
        ordering = ['timestamp']

from django.db import models

class Tenant(models.Model):
    tenant_id = models.IntegerField(primary_key=True)
    tenant_name = models.CharField(max_length=255)
    tenant_kss = models.FileField(upload_to='profiles/', blank=True, null=True)
    chatbot_greeting = models.TextField()
    agent_node_prompt = models.TextField()
    final_answer_prompt = models.TextField()
    summary_prompt = models.TextField()
    tenant_description = models.CharField(max_length=255,blank=True,null=True)

    def __str__(self):
        return self.tenant_name