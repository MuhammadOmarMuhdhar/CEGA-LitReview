import sys
import os
import sys
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
import logging
import traceback


project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from data_fetching import openalex
from classification_algos import few_shot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


aspiration = [
    {"text": "Poverty, Aspirations, and the Economics of Hope'", "label": "Related"},
    {"text": "Poverty, Aspirations and Well-Being: Afraid to Aspire and Unable to Reach a Better Life – Voices from Egypt", "label": "Related"},
    {"text": "Poverty and Aspirations Failure", "label": "Related"},
    {"text": "Hanging in, stepping up and stepping out: livelihood aspirations and strategies of the poor", "label": "Related"},
    {"text": "African migration: trends, patterns, drivers", "label": "Not Related"},
    {"text": "At least he’s doing something’: Moral entrepreneurship and individual responsibility in Jamie’s Ministry of Food", "label": "Not Related"},
    {"text": "Migration and Care: Themes, Concepts and Challenges", "label": "Not Related"},
    {"text": "Whose culture has capital? A critical race theory discussion of community cultural wealth", "label": "Not Related"}
]

mental_health = [
    {"text": "Poverty and Mental Health: How Do Low‐Income Adults and Children Fare in Psychotherapy?", "label": "Related"},
    {"text": "The Impact of Poverty on Mental Health and Wellbeing: A Multi-Level Analysis of Urban Communities", "label": "Related"},
    {"text": "Depression, Anxiety and Stress Among Low-Income Communities: Understanding the Poverty-Mental Health Nexus", "label": "Related"},
    
    {"text": "Advances in Cognitive Behavioral Therapy for Treatment-Resistant Depression", "label": "Not Related"},
    {"text": "The Role of Exercise in Managing Workplace Stress: A Systematic Review", "label": "Not Related"},
    {"text": "Neural Mechanisms of Memory Formation: New Insights from Brain Imaging Studies", "label": "Not Related"}
]

depression = [
    {"text": "Effects of Poverty and Maternal Depression on Early Child Development", "label": "Related"}, 
    {"text": "Depression, Poverty, and Abuse Experience in Suicide Ideation Among Older Koreans", "label": "Related"}, 
    {"text": "Effect of Hippocampal and Amygdala Connectivity on the Relationship Between Preschool Poverty and School-Age Depression", "label": "Related"}, 
    {"text": "Poverty, Depression, and Anxiety: Causal Evidence and Mechanisms Among Low-Income Families", "label": "Related"},

    {"text": "The Rational Optimist: How Prosperity Evolves", "label": "Not Related"}, 
    {"text": "Depression and Anxiety in Relation to Social Status", "label": "Not Related"}, 
    {"text": "Neural Mechanisms of Major Depressive Disorder: Current Evidence and Treatment Implications", "label": "Not Related"},
    {"text": "Cognitive Behavioral Therapy for Treatment-Resistant Depression: A Systematic Review", "label": "Not Related"}
]


anxiety = [
    {"text": "Parenting in poverty: Attention bias and anxiety interact to predict parents’ perceptions of daily parenting hassles.", "label": "Related"},
    {"text": "Problem gambling, anxiety and poverty: an examination of the relationship between poor mental health and gambling problems across socio-economic status", "label": "Related"},
    {"text": "Food insecurity measurement and prevalence estimates during the COVID-19 pandemic in a repeated cross-sectional survey in Mexico", "label": "Related"},
    {"text": "Psychological Distress Among Minority and Low-Income Women Living With HIV", "label": "Related"},
    {"text": "The Great Smoky Mountains Study of Youth", "label": "Not Related"},
    {"text": "Prevalence of Chronic Pain and High-Impact Chronic Pain Among Adults — United States, 2016", "label": "Not Related"},
    {"text": "Lifetime suicidal ideation and suicide attempts in Asian Americans.", "label": "Not Related"},
    {"text": "Operations Research as a Profession", "label": "Not Related"}
]

stress = [
    {"text": "Chernobyl: poverty and stress pose 'bigger threat' than radiation", "label": "Related"},
    {"text": "Poverty-Related Stressors and HIV/AIDS Transmission Risks in Two South African Communities", "label": "Related"},
    {"text": "Socioeconomic status, neighborhood disadvantage, and poverty-related stress: Prospective effects on psychological syndromes among diverse low-income families", "label": "Related"},
    {"text": "Chronic Stress and Economic Hardship: Examining the Effects of Poverty on Psychological Well-being", "label": "Related"},
    {"text": "Resilience as process", "label": "Not Related"},
    {"text": "Stress Management Techniques in the Workplace: A Meta-Analysis", "label": "Not Related"},
    {"text": "The Biology of Stress Response: Neural Mechanisms and Hormonal Changes", "label": "Not Related"},
    {"text": "Exercise and Stress: Impact of Physical Activity on Cortisol Levels", "label": "Not Related"}
]

happiness = [
    {"text": "Does the Risk of Poverty Reduce Happiness?", "label": "Related"},
    {"text": "Poverty, happiness, and risk preferences in rural Ethiopia", "label": "Related"},
    {"text": "Supplemental Material for Wealth, Poverty, and Happiness: Social Class Is Differentially Associated With Positive Emotions", "label": "Related"},
    {"text": "The Relationship Between Income Poverty and Subjective Well-being: Evidence from Rural Communities", "label": "Related"},
    {"text": "The loss of happiness in market democracies", "label": "Not Related"},
    {"text": "Enlightenment Now: The Case for Reason, Science, Humanism, and Progress", "label": "Not Related"},
    {"text": "Health and Happiness among Older Adults", "label": "Not Related"},
    {"text": "The Science of Happiness: Understanding Positive Psychology and Well-being", "label": "Not Related"}
]


beliefs = [
    {"text": "'Rags, Riches, and Bootstraps: Beliefs about the Causes of Wealth and Poverty", "label": "Related"},
    {"text": "The Individual, Society, or Both? A Comparison of Black, Latino, and White Beliefs about the Causes of Poverty", "label": "Related"},
    {"text": "Beliefs About the Causes of Poverty in Parents and Adolescents Experiencing Economic Disadvantage in Hong Kong", "label": "Related"},
    {"text": "Cultural Values and Poverty Attribution: Beliefs about Why People are Poor in Different Societies", "label": "Related"},
    
    {"text": "The Psychology of Religious Belief: Cognitive Approaches and Cultural Impact", "label": "Not Related"},
    {"text": "Understanding Core Beliefs and Cognitive Behavioral Therapy", "label": "Not Related"},
    {"text": "Belief Systems and Social Change: A Theoretical Framework", "label": "Not Related"},
    {"text": "The Role of Cultural Beliefs in Shaping Individual Behavior", "label": "Not Related"}
]

stigma = [
    {"text": "Resisting the Welfare Mother: The Power of Welfare Discourse and Tactics of Resistance", "label": "Related"},
    {"text": "Stigma, discrimination and the health of illicit drug users", "label": "Related"},
    {"text": "Internalized Stigma Among the Poor: Understanding the Psychological Impact of Poverty-Based Discrimination", "label": "Related"},
    {"text": "The Double Burden: Examining the Relationship Between Poverty Stigma and Mental Health Outcomes", "label": "Related"},
    
    {"text": "Mechanisms in the Cycle of Violence", "label": "Not Related"},
    {"text": "The Stigma Complex", "label": "Not Related"},
    {"text": "When White Men Can't Do Math: Necessary and Sufficient Factors in Stereotype Threat", "label": "Not Related"},
    {"text": "Social Identity and Stigma: Understanding Group-Based Prejudice", "label": "Not Related"}
]

mindset = [
    {"text": "The entrepreneurial mindset and poverty", "label": "Related"},
    {"text": "Do mindsets shape intentions to help those in need? Unravelling the paradoxical effects of mindsets of poverty on helping intentions", "label": "Related"},
    {"text": "Scarcity mindset in reproductive health decision making: a qualitative study from rural Malawi", "label": "Related"},
    {"text": "The Effect of Poverty Mindset on Intertemporal Choice: The Mediating Effect of Psychological Capital", "label": "Related"},
    
    {"text": "Sustainable globalization and implications for strategic corporate and national sustainability", "label": "Not Related"},
    {"text": "Work in progress; Enhancing the entrepreneurial mindset of freshman engineers", "label": "Not Related"},
    {"text": "Growth Mindset in Academic Achievement: A Meta-Analytic Review", "label": "Not Related"},
    {"text": "The Role of Mindset in Leadership Development and Organizational Culture", "label": "Not Related"}
]

self_efficacy = [
    {"text": "The impact of poverty on self-efficacy: an Australian longitudinal study", "label": "Related"},
    {"text": "Correlates of food choice in unemployed young people: The role of demographic factors, self-efficacy, food involvement, food poverty and physical activity", "label": "Related"},
    {"text": "Life satisfaction trajectories of junior high school students in poverty: Exploring the role of self‐efficacy and social support", "label": "Related"},
    {"text": "Self-Efficacy and Economic Mobility: How Beliefs Shape Pathways Out of Poverty", "label": "Related"},
    
    {"text": "An Analysis of Outcomes of Reconstruction or Amputation after Leg-Threatening Injuries", "label": "Not Related"},
    {"text": "Why inequality could spread COVID-19", "label": "Not Related"},
    {"text": "Context Matters", "label": "Not Related"},
    {"text": "Self-Efficacy Development in Academic Achievement: A Meta-Analysis of Interventions", "label": "Not Related"}
]

locus_control = [
    # Related examples - diverse mix of locus of control and poverty connections
    {"text": "Locus of control and energy poverty", "label": "Related"},
    {"text": "LOCUS OF CONTROL AND THE ATTRIBUTION FOR POVERTY: COMPARING LEBANESE AND SOUTH AFRICAN UNIVERSITY STUDENTS", "label": "Related"},
    {"text": "Locus of control and culture of poverty. An appraisal of Lawrence M. Mead's ideas in 'Culture and Poverty'", "label": "Related"},
    {"text": "Factors that influence emotional disturbance in adults living in extreme poverty", "label": "Related"},
    {"text": "Welfare Clients' Volunteering as a Means of Empowerment", "label": "Related"},

    # Not Related examples - diverse topics without poverty-locus connection
    {"text": "Anhedonia, Alexithymia and Locus of Control in Unipolar Major Depressive Disorders", "label": "Not Related"},
    {"text": "Genetic and Physiological Analysis of Iron Biofortification in Maize Kernels", "label": "Not Related"},
    {"text": "Rhetorical Structure Theory: Toward a functional theory of text organization", "label": "Not Related"},
    {"text": "Public perception of population health risks in Canada: Risk perception beliefs", "label": "Not Related"},
    {"text": "Urban green space, public health, and environmental justice: The challenge of making cities 'just green enough'", "label": "Not Related"}
]


self_concept = [
    # Related examples - connecting poverty and self-concept
    {"text": "Poverty, Self-Concept, and Health", "label": "Related"},
    {"text": "The Self-Concept of the Poverty Child", "label": "Related"},
    {"text": "Multidimensional analysis procedures for measuring self-concept in poverty area classrooms", "label": "Related"},
    {"text": "Abuse and neglect as predictors of self concept among below poverty line adolescents from India", "label": "Related"},
    {"text": "The 'Inability to be Self‐Reliant' as an Indicator of Poverty: Trends for the U.S., 1975–97", "label": "Related"},

    # Not Related examples - diverse topics without poverty-self-concept connection
    {"text": "Self-Determination Among Mental Health Consumers/Survivors", "label": "Not Related"},
    {"text": "Document Analysis as a Qualitative Research Method", "label": "Not Related"},
    {"text": "Prospect Theory: An Analysis of Decision under Risk", "label": "Not Related"},
    {"text": "Family Therapy: Concepts and Methods", "label": "Not Related"},
    {"text": "Authenticity, Culture and Language Learning", "label": "Not Related"}
]

self_esteem = [
    # Related examples - connecting poverty and self-esteem
    {"text": "'Broken windows' and Self-Esteem: Subjective understandings of neighborhood poverty and disorder", "label": "Related"},
    {"text": "Pathway to neural resilience: Self‐esteem buffers against deleterious effects of poverty on the hippocampus", "label": "Related"},
    {"text": "Pathway of the Association Between Child Poverty and Low Self-Esteem: Results From a Population-Based Study of Adolescents in Japan", "label": "Related"},
    {"text": "Gender and poverty: Self-esteem among elementary school children", "label": "Related"},
    {"text": "Impact of Poverty on Adolescent Drug Use: Moderation Effects of Family Support and Self-Esteem", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-self-esteem connection
    {"text": "Relationships Among Cyberbullying, School Bullying, and Mental Health in Taiwanese Adolescents", "label": "Not Related"},
    {"text": "Lifetime patterns of social phobia: A retrospective study of the course of social phobia in a nonclinical population", "label": "Not Related"},
    {"text": "The family environment in early childhood has a long-term effect on self-esteem: A longitudinal study from birth to age 27 years", "label": "Not Related"},
    {"text": "Effects of Familial Attachment, Social Support, Involvement, and Self-Esteem on Youth Substance Use and Sexual Risk Taking", "label": "Not Related"},
    {"text": "Explanations for unemployment in Britain", "label": "Not Related"}
]

optimism = [
    # Related examples - connecting poverty and optimism
    {"text": "Politics Against Poverty?: Global Pessimism and National Optimism", "label": "Related"},
    {"text": "Halving Poverty by Doubling Aid: Is There Reason for Optimism?", "label": "Related"},
    {"text": "The Influence of Urban Poverty on Students' Academic Optimism: Does Government Assistance Play a Role?", "label": "Related"},
    {"text": "Urban Poverty: From Optimism to Despair and Back Again", "label": "Related"},
    {"text": "Teacher resilience: theorizing resilience and poverty", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-optimism connection
    {"text": "Resource-Conserving Agriculture Increases Yields in Developing Countries", "label": "Not Related"},
    {"text": "Techno-Optimism and Farmers' Attitudes Toward Climate Change Adaptation", "label": "Not Related"},
    {"text": "Internal and External Barriers, Cognitive Style, and the Career Development Variables of Focus and Indecision", "label": "Not Related"},
    {"text": "Trends in Drug Use of Indian Adolescents Living on Reservations: 1975-1983", "label": "Not Related"},
    {"text": "Education: hopes, expectations and achievements of Muslim women in West Yorkshire", "label": "Not Related"}
]
 
cognition_function = [
    # Related examples - connecting poverty and cognition
    {"text": "Poverty Impedes Cognitive Function", "label": "Related"},
    {"text": "Can poverty get under your skin? Basal cortisol levels and cognitive function in children from low and high socioeconomic status", "label": "Related"},
    {"text": "Assessment of Neighborhood Poverty, Cognitive Function, and Prefrontal and Hippocampal Volumes in Children", "label": "Related"},
    {"text": "Duration of Poverty and Subsequent Cognitive Function and Decline Among Older Adults in China, 2005–2018", "label": "Related"},
    {"text": "Family Poverty Affects the Rate of Human Infant Brain Growth", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-cognition connection
    {"text": "Cognitive Deficits Associated with Blood Lead Concentrations <10 microg/dL in US Children and Adolescents", "label": "Not Related"},
    {"text": "Symptoms, cognition, treatment adherence and functional outcome in first-episode psychosis", "label": "Not Related"},
    {"text": "The Positive and Negative Syndrome Scale (PANSS) for Schizophrenia", "label": "Not Related"},
    {"text": "Oral Health and Cognitive Function in the Third National Health and Nutrition Examination Survey (NHANES III)", "label": "Not Related"},
    {"text": "Multiple aspects of self-regulation uniquely predict mathematics but not letter–word knowledge in the early elementary grades", "label": "Not Related"}
]

cognition = [
    # Related examples - connecting poverty and cognition
    {"text": "Linking childhood poverty and cognition: environmental mediators of non‐verbal executive control in an Argentine sample", "label": "Related"},
    {"text": "Heterogeneous Effects of Poverty on Cognition", "label": "Related"},
    {"text": "Achievement gap: Socioeconomic status affects reading development beyond language and cognition in children facing poverty", "label": "Related"},
    {"text": "Connecting Poverty, Culture, and Cognition: The Bridges Out of Poverty Process", "label": "Related"},
    {"text": "Cognitive Deficit and Poverty in the First 5 Years of Childhood in Bangladesh", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-cognition connection
    {"text": "The poverty of embodied cognition", "label": "Not Related"},
    {"text": "Problem solving and computational skill: Are they shared or distinct aspects of mathematical cognition?", "label": "Not Related"},
    {"text": "Moral Nativism: A Sceptical Response", "label": "Not Related"},
    {"text": "The state of emergentism in second language acquisition", "label": "Not Related"},
    {"text": "The History and Philosophy of Ecological Psychology", "label": "Not Related"}
]


cognitive_flexibility = [
    # Related examples - connecting poverty and cognitive flexibility
    {"text": "How Much Does Childhood Poverty Affect the Life Chances of Children?", "label": "Related"},
    {"text": "Socioeconomic status and health: The challenge of the gradient.", "label": "Related"},
    {"text": "Neighborhood Poverty: Context and Consequences for Children", "label": "Related"},
    {"text": "The Lifelong Effects of Early Childhood Adversity and Toxic Stress", "label": "Related"},
    {"text": "Neuroscience, Molecular Biology, and the Childhood Roots of Health Disparities", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-cognitive flexibility connection
    {"text": "Bootstrap Methods and Their Application", "label": "Not Related"},
    {"text": "The Strength of Weak Ties: A Network Theory Revisited", "label": "Not Related"},
    {"text": "Making knowledge the basis of a dynamic theory of the firm", "label": "Not Related"},
    {"text": "Being there: putting brain, body, and world together again", "label": "Not Related"},
    {"text": "The myth of language universals: Language diversity and its importance for cognitive science", "label": "Not Related"}
]

executive_control = [
    # Related examples - connecting poverty and executive control
    {"text": "Income, neural executive processes, and preschool children's executive control", "label": "Related"},
    {"text": "Early childhood poverty and adult executive functioning: Distinct, mediating pathways for different domains of executive functioning", "label": "Related"},
    {"text": "Teacher Stress Predicts Child Executive Function: Moderation by School Poverty", "label": "Related"},
    {"text": "Growing up in poverty and civic engagement: The role of kindergarten executive function and play predicting participation in 8th grade extracurricular activities", "label": "Related"},
    {"text": "Household instability and self-regulation among poor children", "label": "Related"},

    {"text": "Diffusion Tensor Imaging of Frontal White Matter and Executive Functioning in Cocaine-Exposed Children", "label": "Not Related"},
    {"text": "Executive functioning in schizophrenia and the relationship with symptom profile and chronicity", "label": "Not Related"},
    {"text": "Effect of yoga program on executive functions of adolescents dwelling in an orphan home: A randomized controlled study", "label": "Not Related"},
    {"text": "Mindfulness Plus Reflection Training: Effects on Executive Function in Early Childhood", "label": "Not Related"},
    {"text": "Corporate cultures: The rites and rituals of corporate life", "label": "Not Related"}
]

memory = [
    # Related examples - connecting poverty and memory
    {"text": "Childhood poverty is associated with altered hippocampal function and visuospatial memory in adulthood", "label": "Related"},
    {"text": "Working Memory Differences Between Children Living in Rural and Urban Poverty", "label": "Related"},
    {"text": "The role of inflammation in the association between poverty and working memory in childhood", "label": "Related"},
    {"text": "Association between Income and the Hippocampus", "label": "Related"},
    {"text": "The poverty of memory: For political economy in memory studies", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-memory connection
    {"text": "Material‐specific episodic memory associates of the psychomotor poverty syndrome in schizophrenia", "label": "Not Related"},
    {"text": "The Benefits of Reminiscing With Young Children", "label": "Not Related"},
    {"text": "Is the Richness of Our Visual World an Illusion? Transsaccadic Memory for Complex Scenes", "label": "Not Related"},
    {"text": "Memory and Modernity: Popular Culture in Latin America", "label": "Not Related"},
    {"text": "Effects of Early Cerebral Malaria on Cognitive Ability in Senegalese Children", "label": "Not Related"}
]

working_memory = [
    # Related examples - connecting poverty/SES and working memory
    {"text": "Association between poverty, low educational level and smoking with adolescent's working memory: cross lagged analysis from longitudinal data", "label": "Related"},
    {"text": "Early adversity in rural India impacts the brain networks underlying visual working memory", "label": "Related"},
    {"text": "Socioeconomic hardship and delayed reward discounting: Associations with working memory and emotional reactivity", "label": "Related"},
    {"text": "Working Memory Screening, School Context, and Socioeconomic Status", "label": "Related"},
    {"text": "Socioeconomic status is a predictor of neurocognitive performance of early female adolescents", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-working memory connection
    {"text": "Relationship of behavioural and symptomatic syndromes in schizophrenia to spatial working memory and attentional set-shifting ability", "label": "Not Related"},
    {"text": "Visuocognitive Dysfunctions in Parkinson's Disease", "label": "Not Related"},
    {"text": "Art and Agency: An Anthropological Theory", "label": "Not Related"},
    {"text": "The narrative constitution of identity: A relational and network approach", "label": "Not Related"},
    {"text": "Stencil graffiti in urban waterscapes of Buenos Aires and Rosario, Argentina", "label": "Not Related"}
]


fluid_intelligence = [
    # Related examples - connecting poverty and fluid intelligence/cognitive ability
    {"text": "Targeted Estimation of the Relationship Between Childhood Adversity and Fluid Intelligence in a US Population Sample of Adolescents", "label": "Related"},
    {"text": "Childhood poverty: Specific associations with neurocognitive development", "label": "Related"},
    {"text": "Association Between Neighborhood Deprivation and Child Cognition in Clinically Referred Youth", "label": "Related"},
    {"text": "The Psychological Lives of the Poor", "label": "Related"},
    {"text": "Do preschool executive function skills explain the school readiness gap between advantaged and disadvantaged children?", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-intelligence connection
    {"text": "Abilities: Their Structure, Growth, and Action", "label": "Not Related"},
    {"text": "Smart cities of the future", "label": "Not Related"},
    {"text": "Computational Fluid Dynamics for urban physics: Importance, scales, possibilities, limitations and ten tips and tricks towards accurate and reliable simulations", "label": "Not Related"},
    {"text": "The ecological validity of tests of executive function", "label": "Not Related"},
    {"text": "IJER editorial: The future of the internal combustion engine", "label": "Not Related"}
]

attention = [
    # Related examples - connecting poverty and attention
    {"text": "Heterogeneous effects of poverty on attention", "label": "Related"},
    {"text": "Maternal scaffolding and attention regulation in children living in poverty", "label": "Related"},
    {"text": "Sustained attention in infancy: A foundation for the development of multiple aspects of self-regulation for children in poverty", "label": "Related"},
    {"text": "Analysis of attention and analogical reasoning in children of poverty", "label": "Related"},
    {"text": "Poverty and limited attention", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-attention connection
    {"text": "THE ROLE OF PROTECTED AREAS IN CONSERVING BIODIVERSITY AND SUSTAINING LOCAL LIVELIHOODS", "label": "Not Related"},
    {"text": "Scale for the Assessment of Negative Symptoms (SANS)", "label": "Not Related"},
    {"text": "Content Growth and Attention Contagion in Information Networks", "label": "Not Related"},
    {"text": "The political economy of the 'just transition'", "label": "Not Related"},
    {"text": "The possible negative impacts of volunteer tourism", "label": "Not Related"}
]

time_preference = [
    # Related examples - connecting poverty and time preference
    {"text": "Poverty and the Rate of Time Preference: Evidence from Panel Data", "label": "Related"},
    {"text": "Psychological Effects of Poverty on Time Preferences", "label": "Related"},
    {"text": "Poverty Traps and Growth in a Model of Endogenous Time Preference", "label": "Related"},
    {"text": "The Long-Term Effect of Poverty on Time Preference", "label": "Related"},
    {"text": "Effects of Poverty on Impatience: Preferences or Inattention?", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-time preference connection
    {"text": "The General Theory of Employment, Interest and Money", "label": "Not Related"},
    {"text": "Benefits and food safety concerns associated with consumption of edible insects", "label": "Not Related"},
    {"text": "Modern theories of justice", "label": "Not Related"},
    {"text": "The adaptive thermal comfort review from the 1920s, the present, and the future", "label": "Not Related"},
    {"text": "Social Capital: Its Origins and Applications in Modern Sociology", "label": "Not Related"}
]

risk_preference = [
    # Related examples - connecting poverty and risk preferences
    {"text": "Risk preferences and poverty traps in the uptake of credit and insurance amongst small-scale farmers in South Africa", "label": "Related"},
    {"text": "Risk Preferences Under Extreme Poverty: A Field Experiment", "label": "Related"},
    {"text": "Session Discussion: Dynamic Risk Preferences, Poverty Traps, and Thresholds", "label": "Related"},
    {"text": "Risk preference and relative poverty: An analysis based on the data of China Family Panel Studies", "label": "Related"},
    {"text": "Variability in Cross‐Domain Risk Perception among Smallholder Farmers in Mali by Gender and Other Demographic and Attitudinal Characteristics", "label": "Related"},

    # Not Related examples - diverse topics without direct poverty-risk preference connection
    {"text": "Optimal Tax Progressivity: An Analytical Framework", "label": "Not Related"},
    {"text": "Information Feudalism: Who Owns the Knowledge Economy?", "label": "Not Related"},
    {"text": "Sense of community: A definition and theory", "label": "Not Related"},
    {"text": "Happiness economics", "label": "Not Related"},
    {"text": "Social Capital: Implications for Development Theory, Research, and Policy", "label": "Not Related"}
]

def main():
    """Process papers, classify them, and save results."""
    try:
        
        categories = {
            "Affective": ["Poverty and mental health", "Poverty and Depression", "Poverty and Anxiety", 
                         "Poverty and Stress", "Poverty and Happiness"],
            "Beliefs": ["Poverty and Beliefs", "Poverty and Internalized stigma", "Poverty and Mindset", 
                       "Poverty and self-efficacy", "Poverty and locus of control", "Poverty and self concept", 
                       "Poverty and self esteem", "Poverty and Optimism", "Poverty and Aspirations"],
            "Cognitive function": ["Poverty and Cognitive function", "Poverty and Cognition", 
                                 "Poverty and Cognitive flexibility", "Poverty and Executive control", 
                                 "Poverty and Memory", "Poverty and working memory", "Poverty and Fluid intelligence", 
                                 "Poverty and Attention"],
            "Preferences": ["Poverty and Time preference", "Poverty and Risk preference"]
        }

        # Map categories to example datasets
        examples_mapping = {
            "Poverty and mental health": mental_health,
            "Poverty and Depression": depression,
            "Poverty and Anxiety": anxiety,
            "Poverty and Stress": stress,
            "Poverty and Happiness": happiness,
            "Poverty and Beliefs": beliefs,
            "Poverty and Internalized stigma": stigma,
            "Poverty and Mindset": mindset,
            "Poverty and self-efficacy": self_efficacy,
            "Poverty and locus of control": locus_control,
            "Poverty and self concept": self_concept,
            "Poverty and self esteem": self_esteem,
            "Poverty and Optimism": optimism,
            "Poverty and Aspirations": aspiration,
            "Poverty and Cognitive function": cognition_function,
            "Poverty and Cognition": cognition,
            "Poverty and Cognitive flexibility": cognitive_flexibility,
            "Poverty and Executive control": executive_control,
            "Poverty and Memory": memory,
            "Poverty and working memory": working_memory,
            "Poverty and Fluid intelligence": fluid_intelligence,
            "Poverty and Attention": attention,
            "Poverty and Time preference": time_preference,
            "Poverty and Risk preference": risk_preference
        }

        
        # Create output directory at start
        output_dir = "data/sample_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Main progress bar
        with tqdm(total=2, desc="Overall Progress", position=0) as pbar_main:
            # Fetch papers
            data = []
            with tqdm(total=len(sum(categories.values(), [])), desc="Extracting papers", position=1, leave=False) as pbar_extract:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = []
                    for category, subcategories in categories.items():
                        for subcategory in subcategories:
                            future = executor.submit(openalex.extract_papers, category, subcategory, 40)
                            futures.append((future, category, subcategory))
                    
                    for future, category, subcategory in futures:
                        try:
                            papers = future.result()
                            if papers:  # Check if papers is not empty
                                for paper in papers:
                                    paper['category'] = category
                                    paper['keyword'] = subcategory
                                data.extend(papers)
                            pbar_extract.update(1)
                        except Exception as e:
                            logger.error(f"Error processing {category} - {subcategory}: {str(e)}")
                            pbar_extract.update(1)  # Still update progress bar on error

            pbar_main.update(1)

            if not data:
                raise ValueError("No papers were extracted")

            # Process and classify papers
            df = pd.DataFrame(data)
            df.drop_duplicates(subset="doi", inplace=True)
            
            classified_data = {}
            with tqdm(total=len(df['keyword'].unique()), desc="Classifying papers", position=1, leave=False) as pbar_classify:
                for keyword in df['keyword'].unique():
                    keyword_df = df[df['keyword'] == keyword].reset_index(drop=True)
                    try:
                        if not keyword_df.empty and keyword in examples_mapping:
                            classifications = few_shot.classify(
                                texts=keyword_df['title'],
                                examples=examples_mapping[keyword],
                                confidence_threshold=0.2  
                            )
                            
                            keyword_df = pd.concat([keyword_df, classifications], axis=1)
                            classified_data[keyword] = keyword_df
                        
                        pbar_classify.update(1)
                    except Exception as e:
                        logger.error(f"Error classifying {keyword}: {str(e)}")
                        pbar_classify.update(1)

            pbar_main.update(1)

            if not classified_data:
                raise ValueError("No papers were successfully classified")

            # Save results
            final_df = pd.concat(classified_data.values(), ignore_index=True)
            
            # Save full dataset
            final_df.to_csv(os.path.join(output_dir, "full_sample_dataset.csv"), index=False)
            
            # Process related papers
            related_papers = final_df[final_df['label'] == 'Related'].copy()
            
            if not related_papers.empty:
                # Create sub-datasets
                publications = related_papers[['doi', 'title', 'link', 'abstract', 'date', 
                                            'publication', 'field', 'keyword', 'cited_by_count']].copy()
                authors = related_papers[['doi', 'authors']].copy()
                institutions = related_papers[['doi', 'institution', 'country']].copy()
                citations = related_papers[['doi', 'cited_by_count', 'referenced_works', 
                                         'citing_works']].copy()

                # Save sub-datasets
                publications.to_csv(os.path.join(output_dir, "publications.csv"), index=False)
                authors.to_csv(os.path.join(output_dir, "authors.csv"), index=False)
                institutions.to_csv(os.path.join(output_dir, "institutions.csv"), index=False)
                citations.to_csv(os.path.join(output_dir, "citations.csv"), index=False)
            else:
                logger.warning("No related papers were found after classification")
                    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()