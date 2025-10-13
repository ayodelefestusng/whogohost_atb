from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chat_bot import process_message
from .models import Conversation, Message,Tenant
import logging
from django.http import HttpResponse


def get_tenant(tenant_id):
    return tenant_id
# Create your views here.
@csrf_exempt
def chatbot(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'response': 'Invalid request method'}, status=405)

    # Safely extract data from form-data
    user_message = request.POST.get('message', '').strip()
    session_key = request.POST.get('session_key', '').strip()
    attachment = request.FILES.get('attachment', None)  # Optional attachment
    # tenant_prompt = request.FILES.get('tenant_prompt', None)  # Optional attachment
    # tenant_id = request.POST.get('tenant_id', '').strip()
    # tenant_name = request.POST.get('tenant_name', '').strip()
    # tenant_profile = request.FILES.get('tenant_profile', None)  # Optional PDF file
    tenant_id = request.POST.get('tenamt_id', '').strip()
    tenant_name = request.POST.get('tenamt_name', '').strip()
    tenant_kss = request.FILES.get('tenant_profile', None)  # Optional PDF file
    chatbot_greeting = request.POST.get('greeting_prompt', None)  # Optional 
    agent_node_prompt = request.POST.get('agent_node_prompt', None)  # Optional text 
    final_answer_prompt = request.POST.get('final_ouput_prompt', None)  # Optional text 
    summary_prompt = request.POST.get('summary_prompt', None)  # Optional text 
    tenant_description = request.POST.get('additional_note', None)  # Optional text 


    if not tenant_id and not tenant_name:
        return JsonResponse({'status': 'error', 'response': 'Tenant ID and name  are required'}, status=400)

    if not user_message and not attachment:
        return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

    if not session_key:
        return JsonResponse({'status': 'error', 'response': 'Session key is required'}, status=400)
    
    
    if tenant_id:
        tenant, created = Tenant.objects.get_or_create(tenant_id=tenant_id)
        tenant.tenant_name = tenant_name

        if tenant_kss:
            tenant.tenant_kss = tenant_kss.read().decode('utf-8')  # Assuming it's a text file

        if chatbot_greeting:
            tenant.chatbot_greeting = chatbot_greeting  # Intor Line

        if agent_node_prompt:
            tenant.agent_node_prompt = agent_node_prompt  # Agent node prompt
        
        if final_answer_prompt:
            tenant.final_answer_prompt = final_answer_prompt  # Final Answer prompt
        if summary_prompt:
            tenant.summary_prompt = summary_prompt  # optional note

        if tenant_description:
            tenant.tenant_description = tenant_description  # optional note

            tenant.save()

            if created:
                return JsonResponse({"message": "Tenant created successfully."})
            else:
                return JsonResponse({"message": "Tenant updated successfully."})
    else:
        return JsonResponse({"message": "Tenant ID is required."}, status=400)

    # Get or create conversation session
    id = get_tenant(tenant.tenant_id)
    conversation, _ = Conversation.objects.get_or_create(session_id=session_key, is_active=True)

    # Create user message
    user_msg_obj = Message.objects.create(
        conversation=conversation,
        text=user_message,
        is_user=True
    )

  

     # Save attachment if provided
    file_path = ""
    if attachment:
        user_msg_obj.attachment = attachment
        user_msg_obj.save()
        try:
            file_path = user_msg_obj.attachment.path
        except Exception as e:
            logging.warning(f"Could not resolve attachment path: {e}")
            file_path = ""


    # Call your chatbot processor
    try:
        bot_response_data = process_message(user_message, session_key,tenant_id, file_path,)
        bot_metadata = bot_response_data.get('metadata', "I'm sorry, I couldn't process your request.")
    except Exception as e:
        bot_metadata = f"Error processing message: {str(e)}"
        logging.error(f"process_message failed: {e}")

    # Craft the response payload
    response_payload = {
        'status': 'success',
        'response': bot_metadata,
        'attachment_url': user_msg_obj.attachment.url if attachment else None
    }

    logging.info(f"Bot response: {bot_metadata}")
    return JsonResponse(response_payload)


