from calendar import c
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chat_bot import process_message
from .models import Conversation, Message, Tenant
import logging
import os


# ⚠️ REMOVED redundant get_tenant function


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes
from rest_framework.parsers import MultiPartParser, FormParser

from django.conf import settings



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

        # Tenant ID (Required)
        tenant_id = request.POST.get('tenant_id', '').strip()
        tenant_name = request.POST.get('tenant_name', '').strip()

        # User Message (Optional)
        user_message = request.POST.get('user_message', '').strip()
        conversation_id = request.POST.get('conversation_id', '').strip()
        attachment = request.FILES.get('user_msg_attach', None) 

     
     

        #Onbaording requirments
        tenant_faq = request.FILES.get('tenant_faq', None) 
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
        # 3. Always create/update tenant profile
        tenant, created = Tenant.objects.get_or_create(tenant_id=tenant_id)
        tenant.tenant_name = tenant_name

        if tenant_faq:
            try:
                tenant.tenant_faq = tenant_faq
            except Exception as e:
                logging.error(f"Failed to read tenant_kss file: {e}")
                return JsonResponse({'status': 'error', 'response': 'Error processing tenant KSS file'}, status=400)

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

        tenant.save()


        # 4. If no message or summarization, return onboarding success
        if not user_message and not summarization_request:
            return JsonResponse({'status': 'success', 'response': 'Tenant profile updated successfully'})

        # 5. Validate conversation_id for message/summarization
        if not conversation_id:
            return JsonResponse({'status': 'error', 'response': 'Conversation ID is required for message or summarization'}, status=400)

        # 6. Conversation & Message Creation
        conversation, _ = Conversation.objects.get_or_create(conversation_id=conversation_id, is_active=True)

        user_msg_obj = Message.objects.create(
            conversation=conversation,
            text=user_message,
            is_user=True
        )

    # 7. Attachment Handling
 
        file_path = ""
        if attachment:
            user_msg_obj.attachment = attachment
            user_msg_obj.save()
            try:
                file_path = user_msg_obj.attachment.path
            except Exception as e:
                logging.warning(f"Could not resolve attachment path: {e}")
                file_path = ""

        # 8. Call Chatbot Processor

        try:
            # Passed tenant_id to process_message
            bot_response_data = process_message(user_message, conversation_id, tenant_id, file_path,summarization_request)
            bot_metadata = bot_response_data.get('metadata', "I'm sorry, I couldn't process your request.")
            bot_answer = bot_response_data.get('answer', "I'm sorry, I couldn't process your request.")
           

            if user_message:
                response = bot_answer
            elif bot_metadata:
                response = bot_metadata
            else:
                response = bot_response_data.get('answer', bot_metadata)


            print ("Bot response Akula:", response)
        except Exception as e:
            bot_metadata = f"Error processing message: {str(e)}"
            logging.error(f"process_message failed: {e}")

        # 9. Craft and Return Response

        response_payload = {
            'status': 'success',
            'response': response,
            'Tenant_id': tenant_id,
            'conversation_id': conversation_id,

            # 'attachment_url': user_msg_obj.attachment.url if attachment else None
        }

        logging.info(f"Bot response: {response} from tenannt_id{tenant_id} for conv_id {conversation_id} ")
        return JsonResponse(response_payload)

@method_decorator(csrf_exempt, name='dispatch')
class Onboard(APIView):

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

        # Tenant ID (Required)
        tenant_id = request.POST.get('tenant_id', '').strip()
        tenant_name = request.POST.get('tenant_name', '').strip()

        # User Message (Optional)
        # user_message = request.POST.get('user_message', '').strip()
        # conversation_id = request.POST.get('conversation_id', '').strip()
        # attachment = request.FILES.get('user_msg_attach', None) 

     
     

        #Onbaording requirments
        tenant_faq = request.FILES.get('tenant_faq', None) 
        tenant_mandate = request.FILES.get('tenant_mandate', None) 
        chatbot_greeting = request.POST.get('chat_bot_intro', None) 
        agent_node_prompt = request.POST.get('agent_node_prompt', None)
        tenant_description = request.POST.get('additional_note', None)
        final_answer_prompt = request.POST.get('output_prompt', None)
        summary_prompt = request.POST.get('summary_prompt', None)

        #Summarisation Request
        # summarization_request = request.POST.get('summarisation_request', None)
        

        # 2. Validation
        if not tenant_id or not tenant_name: # Use 'or' for simple check
            return JsonResponse({'status': 'error', 'response': 'Tenant ID and name are required'}, status=400)
        # 3. Always create/update tenant profile

        # 3. Check if tenant already exists
        existing_tenant = Tenant.objects.filter(tenant_id=tenant_id).first()
        if existing_tenant:
            return JsonResponse({'status': 'error', 'response': 'Tenant ID already exists'}, status=400)

        # 4. Create tenant profile

        tenant, created = Tenant.objects.get_or_create(tenant_id=tenant_id)
        tenant.tenant_name = tenant_name

        if tenant_faq:
            try:
                tenant.tenant_faq = tenant_faq
            except Exception as e:
                logging.error(f"Failed to read tenant_faq file: {e}")
                return JsonResponse({'status': 'error', 'response': 'Error processing tenant FAQs file'}, status=400)
        if tenant_mandate:
            try:
                tenant.tenant_mandate = tenant_mandate
            except Exception as e:
                logging.error(f"Failed to read tenant_mandate file: {e}")
                return JsonResponse({'status': 'error', 'response': 'Error processing tenant Mandate file'}, status=400)
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

        tenant.save()
        return JsonResponse({'status': 'success', 'response': 'Tenant profile created successfully'})


@method_decorator(csrf_exempt, name='dispatch')
class Update(APIView):
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
        description="Update tenant profile with onboarding and configuration data"
    )
    def post(self, request):
        tenant_id = request.data.get('tenant_id', '').strip()
        tenant_name = request.data.get('tenant_name', '').strip()
        tenant_faq = request.FILES.get('tenant_faq')
        tenant_mandate = request.FILES.get('tenant_mandate', None)

        if not tenant_id:
            return JsonResponse({'status': 'error', 'response': 'Tenant ID is required'}, status=400)
        if not tenant_faq and not tenant_mandate:
            return JsonResponse({'status': 'error', 'response': 'Tenant Mandate or FAQs is required'}, status=400)

        try:
            tenant = Tenant.objects.get(tenant_id=tenant_id)
        except Tenant.DoesNotExist:
            return JsonResponse({'status': 'error', 'response': 'Tenant ID does not exist'}, status=404)

        tenant.tenant_name = tenant_name

        if tenant_faq:
            tenant.tenant_faq = tenant_faq
        if tenant_mandate:
            tenant.tenant_mandate = tenant_mandate

        optional_fields = {
            'chatbot_greeting': request.data.get('chat_bot_intro'),
            'agent_node_prompt': request.data.get('agent_node_prompt'),
            'final_answer_prompt': request.data.get('output_prompt'),
            'summary_prompt': request.data.get('summary_prompt'),
            'tenant_description': request.data.get('additional_note'),
        }

        for field, value in optional_fields.items():
            if value:
                setattr(tenant, field, value)

        tenant.save()
        return JsonResponse({'status': 'success', 'response': 'Tenant profile updated successfully'}, status=200)

@method_decorator(csrf_exempt, name='dispatch')
class FetchDocument(APIView):
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        methods=["GET"],
        parameters=[
            OpenApiParameter(name='tenant_id', type=OpenApiTypes.STR, required=True, location=OpenApiParameter.QUERY),
        ],
        description="Chatbot endpoint for onboarding, messaging, and summarization"
    )
    def get(self, request):
        tenant_id = request.GET.get('tenant_id', '').strip()
        print("Akkdk", tenant_id)

        if not tenant_id:
            return JsonResponse({'status': 'error', 'response': 'Tenant ID is required'}, status=400)

        try:
            tenant = Tenant.objects.get(tenant_id=tenant_id)
        except Tenant.DoesNotExist:
            return JsonResponse({'status': 'error', 'response': 'Tenant ID does not exist'}, status=404)

        faq = tenant.tenant_faq
        mandate = tenant.tenant_mandate
        faq_url = request.build_absolute_uri(faq.url) if faq else None
        mandate_url = request.build_absolute_uri(mandate.url) if mandate else None
        if faq and mandate:
            return JsonResponse({
                'status': 'success',
                'response': {
                    'faq_url': request.build_absolute_uri(faq.url),
                    'faq_name': os.path.basename(faq.name),
                    'mandate_url': request.build_absolute_uri(mandate.url),
                    'mandate_name': os.path.basename(mandate.name)
                }
            }, status=200)
        elif faq:
            return JsonResponse({
                'status': 'success',
                'response': {
                    'file_type': 'FAQ',
                    'file_url': request.build_absolute_uri(faq.url),
                    'file_name': os.path.basename(faq.name)
                }
            }, status=200)
        elif mandate:
            return JsonResponse({
                'status': 'success',
                'response': {
                    'file_type': 'Mandate',
                    'file_url': request.build_absolute_uri(mandate.url),
                    'file_name': os.path.basename(mandate.name)
                }
            }, status=200)

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
    tenant_kss_file = request.FILES.get('tenant_faq', None) 
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
                return JsonResponse({'status': 'error', 'response': 'Error processing tenant FAQs file'}, status=400)

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
    
        return JsonResponse({'status': 'success', 'response': 'Tenant profile created/updated successfully'}, )

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
        bot_answer = bot_response_data.get('metadata', "I'm sorry, I couldn't process your request.")
        response = ""

        if user_message:
            response = bot_answer
        elif bot_metadata:
            response = bot_metadata
        else:
            response = bot_response_data.get('answer', bot_metadata)


        print ("Bot response Akula:", response)
    except Exception as e:
        bot_metadata = f"Error processing message: {str(e)}"
        logging.error(f"process_message failed: {e}")

    # 7. Craft and Return Response
    response_payload = {
        'status': 'success',
        'response': bot_metadata,
        'Tenant_id': tenant_id,
        'conversation_id': conversation_id,

        # 'attachment_url': user_msg_obj.attachment.url if attachment else None
    }

    logging.info(f"Bot response: {response} from tenannt_id{tenant_id} for conv_id {conversation_id} ")
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


# from .models import Prompt,Prompt7
import os

from django.conf import settings
from dotenv import load_dotenv
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
import json
import logging

from .payroll import desire,aluke,important,get_payslips_from_json,atb,parse_markdown_to_json,systemprompt



load_dotenv()


# # Set environment variables for LangSmith
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_PAYROLL")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY if LANGSMITH_API_KEY else ""
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT if LANGSMITH_PROJECT else "Agent_Creation"
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT if LANGSMITH_ENDPOINT else "https://api.smith.langchain.com"

google_model="gemini-flash-latest"

llmv = ChatGoogleGenerativeAI(model=google_model, temperature=0, google_api_key=GOOGLE_API_KEY)    
# llmv=ChatGroq(model ="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0,max_tokens=None,timeout=None,max_retries=2)




desired_columns = desire()





@csrf_exempt
def variance(request):
    print("Variance endpoint called")

    if request.method != 'POST':
        return JsonResponse({
            'status': 'error',
            'response': 'Please submit via POST with two files: "old" and "new".'
        }, status=405)

    raw_old = request.FILES.get('old')
    raw_new = request.FILES.get('new')

    if not raw_old or not raw_new:
        return JsonResponse({'status': 'error', 'response': 'Missing files'}, status=400)

    key_fields = important()
    print("Variance endpoint MIDDLE ")
    

    try:
        old = get_payslips_from_json(raw_old,desired_columns)
        new = get_payslips_from_json(raw_new,desired_columns)

#         old = {
#   "employees": [
#     {"id": "emp001", "name": "Alice", "gross": 200000, "tax": 10000, "pension": 8000},
#     {"id": "emp002", "name": "Bob", "gross": 180000, "tax": 9000, "pension": 7200}
#   ]
# }
#         new = {
#   "employees": [
#     {"id": "emp001", "name": "Alice", "gross": 220000, "tax": 11000, "pension": 8800},
#     {"id": "emp003", "name": "Charlie", "gross": 190000, "tax": 9500, "pension": 7600}
#   ]
# }


        
        # datar = json.load(old)
        # datat = json.load(new)
       
        # payslips_dfr = pd.json_normalize(datar["payslips"])
        # payslips_dft = pd.json_normalize(datat["payslips"])

        # initial_json = payslips_dfr[key_fields].to_json(orient="records", indent=4)
        # treated_json = payslips_dft[key_fields].to_json(orient="records", indent=4)
        # py2 = Prompt7.objects.get(pk=1)  # Get the existing record

        # retrieved_template6 = py2.variance_prompt
        retrieved_template6 = systemprompt(old, new)


        # result = atb(old, new,llmv,retrieved_template6)
        result_markdown = atb(old, new, llmv, retrieved_template6)
        import re

        # Remove Markdown code block if present
        if result_markdown.strip().startswith("```json"):
            result_markdown = re.sub(r"^```json\s*", "", result_markdown.strip())
            result_markdown = re.sub(r"\s*```$", "", result_markdown.strip())

        # print('ye,m',result_markdown)
        # structured_data = parse_markdown_to_json(result_markdown)
        try:
            structured_data = json.loads(result_markdown)
        except json.JSONDecodeError as e:
            return JsonResponse({'status': 'error', 'response': f'Invalid JSON format: {str(e)}'}, status=500)




        if request.GET.get("format") == "html":
            print("Variance endpoint successful")
            return render(request, 'myapp/variance_result.html', {
                'status': 'success',
                'data': structured_data
                
            })
        

        print("Variance endpoint successful")
        return JsonResponse({'status': 'success', 'data': structured_data}, status=200)

    except Exception as e:
        print("Variance endpoint failed")
        return JsonResponse({'status': 'error', 'response': str(e)}, status=500)
    
from django.shortcuts import render


def variance_upload_form(request):
    return render(request, 'myapp/variance_upload.html')  # This matches your file path


def hello_json(request):
    data = {"message": "Hello, world!"}
    return JsonResponse(data)
