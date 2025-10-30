from calendar import c
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chat_bot import process_message
from .models import Conversation, Message, Tenant
import logging

# ⚠️ REMOVED redundant get_tenant function


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes


@method_decorator(csrf_exempt, name='dispatch')
class ChatbotView(APIView):

    # @extend_schema(
    #     request=None,  # You can define a serializer here for request body
    #     responses={200: None},  # You can define a serializer here for response
    #     methods=["POST"],
    #     description="Chatbot endpoint for onboarding, messaging, and summarization"
    # )
    # Set parser for file uploads
    from rest_framework.parsers import MultiPartParser, FormParser
    parser_classes = [MultiPartParser, FormParser]
    @extend_schema(
        methods=["POST"],
        parameters=[
            OpenApiParameter(name='tenant_id', type=OpenApiTypes.STR, required=True, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='tenant_name', type=OpenApiTypes.STR, required=True, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='user_message', type=OpenApiTypes.STR, required=False, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='conversation_id', type=OpenApiTypes.STR, required=False, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='chat_bot_intro', type=OpenApiTypes.STR, required=False, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='agent_node_prompt', type=OpenApiTypes.STR, required=False, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='output_prompt', type=OpenApiTypes.STR, required=False, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='summary_prompt', type=OpenApiTypes.STR, required=False, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='additional_note', type=OpenApiTypes.STR, required=False, location=OpenApiParameter.QUERY),
            OpenApiParameter(name='summarisation_request', type=OpenApiTypes.STR, required=False, location=OpenApiParameter.QUERY),
        ],
        description="Chatbot endpoint for onboarding, messaging, and summarization"
    )



    def post(self, request):
        
        print("Chatbot view called")
        print("GET params:", request.GET)
        print("POST params:", request.POST)
        print("FILES:", request.FILES)
        if request.method != 'POST':
            return JsonResponse({'status': 'error', 'response': 'Invalid request method'}, status=405)

        # 1. Extract Data
       
        user_message = request.POST.get('user_message', '').strip()
        conversation_id = request.POST.get('conversation_id', '').strip()
        attachment = request.FILES.get('user_msg_attach', None) 

        # Corrected spellings in the request extraction to match the rest of your code
        tenant_id = request.POST.get('tenant_id', '').strip()
        tenant_name = request.POST.get('tenant_name', '').strip()
        print ("Request Data:", tenant_name)
        print("User Message:", user_message)
        print("Conversation ID:", conversation_id)
        print("Tenant ID:", tenant_id)
        print("Tenant Name:", tenant_name)


        # tenant_id = request.POST.get('tenant_id') or request.GET.get('tenant_id', '')
        # tenant_name = request.POST.get('tenant_name') or request.GET.get('tenant_name', '')

        #Onbaording requirments
        tenant_kss_file = request.FILES.get('tenant_profile', None) 
        chatbot_greeting = request.POST.get('chat_bot_intro', None) 
        agent_node_prompt = request.POST.get('agent_node_prompt', None)
        tenant_description = request.POST.get('additional_note', None)
        final_answer_prompt = request.POST.get('output_prompt', None)
        summary_prompt = request.POST.get('summary_prompt', None)

        #Summarisation Request
        summarization_request = request.POST.get('summarisation_request', None)
        

        # 2. Validation
        if not tenant_id or not tenant_name: # Use 'or' for simple check
            return JsonResponse({'status': 'error', 'response': 'Tenant ID and name are required'}, status=400)


        # ... (Validation for tenant_id and tenant_name)

    # **New Section: Handle Tenant Configuration/Onboarding**
    # This block executes if it's purely a configuration request (no message/summary request)
        print ("User Message:", user_message, "Summarisation Request:", summarization_request)
        is_onboarding_request = not user_message and not summarization_request

        if is_onboarding_request:
            tenant, created = Tenant.objects.get_or_create(tenant_id=tenant_id)
            tenant.tenant_name = tenant_name

            if tenant_kss_file:
                # Assuming the field on the model is named 'tenant_kss'
                # NOTE: Your model field should be a TextField if you are doing .read().decode()
                try:
                    # tenant.tenant_kss = tenant_kss_file.read().decode('utf-8') 
                    tenant.tenant_kss = tenant_kss_file
                except Exception as e:
                    logging.error(f"Failed to read tenant_kss file: {e}")
                    return JsonResponse({'status': 'error', 'response': 'Error processing tenant KSS file'}, status=400)

            # Update other fields if they exist in the request
            if chatbot_greeting:
                tenant.chatbot_greeting = chatbot_greeting
            if agent_node_prompt:
                tenant.agent_node_prompt = agent_node_prompt
            if final_answer_prompt:
                tenant.final_answer_prompt = final_answer_prompt
            if summary_prompt:
                tenant.summary_prompt = summary_prompt
            if tenant_description:
                tenant.tenant_description = tenant_description 

            # Save all updates to the Tenant object
            tenant.save() 
        
            return JsonResponse({'status': 'success', 'response': 'Tenant profile updated successfully'}, )

        # **New Section: Handle Chat/Summarization Request**
        # This block executes for chat or summarization requests
        if (user_message or summarization_request) and not conversation_id:
            return JsonResponse({'status': 'error', 'response': 'conversation id  is required'}, status=400)

        # 4. Conversation & Message Creation  session_id
        conversation, _ = Conversation.objects.get_or_create(conversation_id=conversation_id, is_active=True)

        # Create user message
        user_msg_obj = Message.objects.create(
            conversation=conversation,
            text=user_message,
            is_user=True
        )

        # 5. Attachment Handling
        file_path = ""
        if attachment:
            user_msg_obj.attachment = attachment
            user_msg_obj.save()
            try:
                file_path = user_msg_obj.attachment.path
            except Exception as e:
                logging.warning(f"Could not resolve attachment path: {e}")
                file_path = ""

        # 6. Call Chatbot Processor
        try:
            # Passed tenant_id to process_message
            bot_response_data = process_message(user_message, conversation_id, tenant_id, file_path,summarization_request)
            bot_metadata = bot_response_data.get('metadata', "I'm sorry, I couldn't process your request.")
            
        except Exception as e:
            bot_metadata = f"Error processing message: {str(e)}"
            logging.error(f"process_message failed: {e}")

        # 7. Craft and Return Response
        response_payload = {
            'status': 'success',
            'response':bot_metadata,
            'attachment_url': user_msg_obj.attachment.url if attachment else None
        }

        logging.info(f"Bot response: {bot_metadata}")
        # return JsonResponse(response_payload)

            # Paste your chatbot logic here
      
        return Response(response_payload, status=status.HTTP_200_OK)



@csrf_exempt
def chatbot2(request):
    print("Chatbot view called")
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'response': 'Invalid request method'}, status=405)

    # 1. Extract Data
    user_message = request.POST.get('user_message', '').strip()
    conversation_id = request.POST.get('conversation_id', '').strip()
    attachment = request.FILES.get('user_msg_attach', None) 

    # Corrected spellings in the request extraction to match the rest of your code
    # tenant_id = request.POST.get('tenamt_id', '').strip() 
    tenant_id = request.POST.get('tenant_id', '').strip()
    tenant_name = request.POST.get('tenant_name', '').strip()

    #Onbaording requirments
    tenant_kss_file = request.FILES.get('tenant_profile', None) 
    chatbot_greeting = request.POST.get('chat_bot_intro', None) 
    agent_node_prompt = request.POST.get('agent_node_prompt', None)
    tenant_description = request.POST.get('additional_note', None)
    final_answer_prompt = request.POST.get('output_prompt', None)
    summary_prompt = request.POST.get('summary_prompt', None)

    #Summarisation Request
    summarization_request = request.POST.get('summarisation_request', None)
     

    # 2. Validation
    if not tenant_id or not tenant_name: # Use 'or' for simple check
        return JsonResponse({'status': 'error', 'response': 'Tenant ID and name are required'}, status=400)


    # ... (Validation for tenant_id and tenant_name)

# **New Section: Handle Tenant Configuration/Onboarding**
# This block executes if it's purely a configuration request (no message/summary request)
    is_onboarding_request = not user_message and not summarization_request

    if is_onboarding_request:
        tenant, created = Tenant.objects.get_or_create(tenant_id=tenant_id)
        tenant.tenant_name = tenant_name

        if tenant_kss_file:
            # Assuming the field on the model is named 'tenant_kss'
            # NOTE: Your model field should be a TextField if you are doing .read().decode()
            try:
                # tenant.tenant_kss = tenant_kss_file.read().decode('utf-8') 
                tenant.tenant_kss = tenant_kss_file
            except Exception as e:
                logging.error(f"Failed to read tenant_kss file: {e}")
                return JsonResponse({'status': 'error', 'response': 'Error processing tenant KSS file'}, status=400)

        # Update other fields if they exist in the request
        if chatbot_greeting:
            tenant.chatbot_greeting = chatbot_greeting
        if agent_node_prompt:
            tenant.agent_node_prompt = agent_node_prompt
        if final_answer_prompt:
            tenant.final_answer_prompt = final_answer_prompt
        if summary_prompt:
            tenant.summary_prompt = summary_prompt
        if tenant_description:
            tenant.tenant_description = tenant_description 

        # Save all updates to the Tenant object
        tenant.save() 
    
        return JsonResponse({'status': 'success', 'response': 'Tenant profile updated successfully'}, )

    # **New Section: Handle Chat/Summarization Request**
    # This block executes for chat or summarization requests
    if (user_message or summarization_request) and not conversation_id:
        return JsonResponse({'status': 'error', 'response': 'conversation id  is required'}, status=400)

    # 4. Conversation & Message Creation  session_id
    conversation, _ = Conversation.objects.get_or_create(conversation_id=conversation_id, is_active=True)

    # Create user message
    user_msg_obj = Message.objects.create(
        conversation=conversation,
        text=user_message,
        is_user=True
    )

    # 5. Attachment Handling
    file_path = ""
    if attachment:
        user_msg_obj.attachment = attachment
        user_msg_obj.save()
        try:
            file_path = user_msg_obj.attachment.path
        except Exception as e:
            logging.warning(f"Could not resolve attachment path: {e}")
            file_path = ""

    # 6. Call Chatbot Processor
    try:
        # Passed tenant_id to process_message
        bot_response_data = process_message(user_message, conversation_id, tenant_id, file_path,summarization_request)
        bot_metadata = bot_response_data.get('metadata', "I'm sorry, I couldn't process your request.")
        print ("Bot response Akula:", bot_response_data.get('answer'))
    except Exception as e:
        bot_metadata = f"Error processing message: {str(e)}"
        logging.error(f"process_message failed: {e}")

    # 7. Craft and Return Response
    response_payload = {
        'status': 'success',
        'response': bot_metadata,
        'attachment_url': user_msg_obj.attachment.url if attachment else None
    }

    logging.info(f"Bot response: {bot_metadata}")
    return JsonResponse(response_payload)




@csrf_exempt
def chatbot1(request):
    # if request.method != 'POST':
    #     return JsonResponse({'status': 'error', 'response': 'Invalid request method'}, status=405)

    # 1. Extract Data
    # user_message = request.POST.get('message', '').strip()
    # session_key = request.POST.get('session_key', '').strip()
    # attachment = request.FILES.get('attachment', None) 


 
    
    # Corrected spellings in the request extraction to match the rest of your code
    tenant_id = request.POST.get('tenamt_id', '').strip() 
    tenant_name = request.POST.get('tenamt_name', '').strip()


    user_message = "Can I get a loan and savings "
    session_key = "12346877k"
    tenant_id ="1"
    tenant_name = "ATBs"
    attachment = request.FILES.get('attachment', None) 
    

    tenant_kss_file = request.FILES.get('tenant_profile', None) 
    chatbot_greeting = request.POST.get('greeting_prompt', None) 
    agent_node_prompt = request.POST.get('agent_node_prompt', None)
    final_answer_prompt = request.POST.get('final_ouput_prompt', None)
    summary_prompt = request.POST.get('summary_prompt', None)
    tenant_description = request.POST.get('additional_note', None) 

    # 2. Validation
    if not tenant_id or not tenant_name: # Use 'or' for simple check
        return JsonResponse({'status': 'error', 'response': 'Tenant ID and name are required'}, status=400)

    if not user_message and not attachment:
        return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

    if not session_key:
        return JsonResponse({'status': 'error', 'response': 'Session key is required'}, status=400)
    
    
    # 3. Tenant Management (Refactored to save/update and continue)
    tenant, created = Tenant.objects.get_or_create(tenant_id=tenant_id)
    tenant.tenant_name = tenant_name

    if tenant_kss_file:
        # Assuming the field on the model is named 'tenant_kss'
        # NOTE: Your model field should be a TextField if you are doing .read().decode()
        try:
            tenant.tenant_kss = tenant_kss_file.read().decode('utf-8') 
        except Exception as e:
            logging.error(f"Failed to read tenant_kss file: {e}")
            return JsonResponse({'status': 'error', 'response': 'Error processing tenant KSS file'}, status=400)

    # Update other fields if they exist in the request
    if chatbot_greeting:
        tenant.chatbot_greeting = chatbot_greeting
    if agent_node_prompt:
        tenant.agent_node_prompt = agent_node_prompt
    if final_answer_prompt:
        tenant.final_answer_prompt = final_answer_prompt
    if summary_prompt:
        tenant.summary_prompt = summary_prompt
    if tenant_description:
        tenant.tenant_description = tenant_description 

    # Save all updates to the Tenant object
    tenant.save() 
    
    # 4. Conversation & Message Creation
    conversation, _ = Conversation.objects.get_or_create(session_id=session_key, is_active=True)

    # Create user message
    user_msg_obj = Message.objects.create(
        conversation=conversation,
        text=user_message,
        is_user=True
    )

    # 5. Attachment Handling
    file_path = ""
    if attachment:
        user_msg_obj.attachment = attachment
        user_msg_obj.save()
        try:
            file_path = user_msg_obj.attachment.path
        except Exception as e:
            logging.warning(f"Could not resolve attachment path: {e}")
            file_path = ""

    # 6. Call Chatbot Processor
    try:
        # Passed tenant_id to process_message
        bot_response_data = process_message(user_message, session_key, tenant_id, file_path)
        bot_metadata = bot_response_data.get('metadata', "I'm sorry, I couldn't process your request.")
    except Exception as e:
        bot_metadata = f"Error processing message: {str(e)}"
        logging.error(f"process_message failed: {e}")

    # 7. Craft and Return Response
    response_payload = {
        'status': 'success',
        'response': bot_metadata,
        'attachment_url': user_msg_obj.attachment.url if attachment else None
    }

    logging.info(f"Bot response: {bot_metadata}")
    return JsonResponse(response_payload)

